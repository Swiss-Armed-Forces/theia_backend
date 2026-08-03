from __future__ import annotations
import copy
import datetime
import json
from typing import Literal
import numpy as np
import pydantic

from theia.config import TERRAIN_HBV_DATA_DIR
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.effectors import DirectFireEffector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.fixed_path_kamikaze_drone import FixedPathOneWayDrone
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.damage_model import AbstractDamageModel, UniformDamageModel
from theia.simulation.simulator import Simulator, TimeCriterion
from theia.simulation.theia_logging import (
    CompositeSimulationListener,
    FileLogger,
    SituationalPictureBuffer,
)
from theia.simulation.trackers.pseudo_tracker import PseudoTracker
from theia.terrain import AbstractTerrainModel, SrtmTerrainModel
from theia.terrain_fast_los import FastSrtmModel, HbvTree
from theia.types import (
    AbstractTracker,
    ConstantRcsModel,
    Controller,
    Entity,
    IdProvider,
    MonostaticSensor,
    PclSensor,
    Point,
    Trajectory,
)


class TerrainFactory(pydantic.BaseModel):
    terrain_name: str

    _terrain_models: pydantic.ClassVar[dict[str, AbstractTerrainModel]] = {}

    def to_terrain(self) -> AbstractTerrainModel:
        # Avoid loading the same terrain twice.
        if self.terrain_name in self._terrain_models:
            return self._terrain_models[self.terrain_name]

        if self.terrain_name == "SRTM":
            terrain = SrtmTerrainModel()
        elif self.terrain_name.startswith("tree_"):
            tree = HbvTree.load(f"{TERRAIN_HBV_DATA_DIR}/{self.terrain_name}.zip")
            terrain = FastSrtmModel(tree=tree, srtm_model=SrtmTerrainModel())
        else:
            raise ValueError(f"Unknown terrain model {self.terrain_name}")

        self._terrain_models[self.terrain_name] = terrain
        return terrain


class DirectFireEffectorFactory(pydantic.BaseModel):
    id: int
    """Effector ID"""
    name: str
    """Human-readable name for this effector"""
    point: Point
    """Position of the effector."""
    combat_range: float
    """Up to which distance a target can be fought [m]"""
    n_attacks_left: int
    """Number of attacks the effector has left"""
    terrain: TerrainFactory

    def to_effector(self) -> DirectFireEffector:
        return DirectFireEffector(
            id=self.id,
            name=self.name,
            point=self.point,
            combat_range=self.combat_range,
            n_attacks_left=self.n_attacks_left,
            terrain=self.terrain.to_terrain(),
        )


class FixedPathOneWayDroneFactory(pydantic.BaseModel):
    effector: DirectFireEffectorFactory
    trajectory: Trajectory
    assigned_goal: Point

    def to_controller(self, terrain: AbstractTerrainModel) -> Controller:
        return FixedPathOneWayDrone(
            effector=self.effector.to_effector(),
            trajectory=self.trajectory,
            assigned_goal=self.assigned_goal,
            terrain=terrain,
        )


class MobileDispositive(pydantic.BaseModel):
    oneway_drones: list[FixedPathOneWayDroneFactory]

    def to_controller(self, terrain: AbstractTerrainModel) -> Controller:
        oneway_drone_controllers = [
            drone.to_controller(terrain) for drone in self.oneway_drones
        ]
        return ControllerGroup(controllers=oneway_drone_controllers)

    @staticmethod
    def from_file(file: str) -> MobileDispositive:
        with open(file, "r") as file:
            return MobileDispositive.model_validate_json(file.read())

    def update_id_provider(self, id_provider: IdProvider):
        for drone in self.oneway_drones:
            id_provider.register_entity(Entity.EFFECTOR, drone.effector.id)
            id_provider.register_entity(Entity.TARGET, drone.trajectory.target_id)

    @property
    def t_min(self) -> datetime.datetime | None:
        if len(self.oneway_drones) == 0:
            return None
        return min([drone.trajectory.times[0] for drone in self.oneway_drones])

    @property
    def t_max(self) -> datetime.datetime:
        if len(self.oneway_drones) == 0:
            return None
        return max([drone.trajectory.times[-1] for drone in self.oneway_drones])


class StaticDispositive(pydantic.BaseModel):
    monostatic_sensors: list[MonostaticSensor]
    pcl_sensors: list[PclSensor]
    effectors: list[DirectFireEffector]

    def to_controller(
        self,
        monostatic_sensor_rcs: float,
        id_provider: IdProvider,
        is_blue: bool,
    ) -> Controller:
        monostatic_controllers = [
            MonostaticRadarController(
                # TODO: Ensure that the target ID is still consistent after adding actual targets!
                target_id=id_provider.increment(Entity.TARGET),
                radar=sensor,
                is_blue=is_blue,
                rcs_model=ConstantRcsModel(rcs=monostatic_sensor_rcs),
            )
            for sensor in self.monostatic_sensors
        ]

        # TODO: Load PCL sensors.
        # TODO: Load effectors.
        pass

        return ControllerGroup(controllers=monostatic_controllers)

    @staticmethod
    def from_file(static_dispositive_file: str) -> StaticDispositive:
        with open(static_dispositive_file, "r") as file:
            data = json.load(file)
            dispo = StaticDispositive(
                monostatic_sensors=[
                    MonostaticSensor.model_validate(s)
                    for s in data["monostatic_sensors"]
                ],
                pcl_sensors=[PclSensor.model_validate(s) for s in data["pcl_sensors"]],
                effectors=[
                    DirectFireEffectorFactory.model_validate(e).to_effector()
                    for e in data["effectors"]
                ],
            )

        return dispo

    def update_id_provider(self, id_provider: IdProvider):
        for sensor in self.monostatic_sensors:
            id_provider.register_entity(Entity.SENSOR, sensor.id)
            id_provider.register_entity(Entity.TRANSMITTER, sensor.transmitter.id)
            id_provider.register_entity(Entity.RECEIVER, sensor.receiver.id)
        for sensor in self.pcl_sensors:
            if sensor.receiver.id in id_provider._used_ids[Entity.RECEIVER]:
                raise ValueError(f"PCL Receiver ID duplicated: {sensor.receiver.id}")
            if sensor.transmitter.id in id_provider._used_ids[Entity.TRANSMITTER]:
                raise ValueError(
                    f"PCL Transmitter ID duplicated: {sensor.transmitter.id}"
                )
        for sensor in self.pcl_sensors:
            id_provider.register_entity(Entity.SENSOR, sensor.id)
            # Receiver and Transmitter can be duplicated in PCL sensors.
            try:
                id_provider.register_entity(Entity.TRANSMITTER, sensor.transmitter.id)
            except ValueError:
                pass
            try:
                id_provider.register_entity(Entity.RECEIVER, sensor.receiver.id)
            except ValueError:
                pass
        for effector in self.effectors:
            id_provider.register_entity(Entity.EFFECTOR, effector.id)

    @staticmethod
    def merge(
        deployment1: StaticDispositive,
        deployment2: StaticDispositive,
    ) -> StaticDispositive:
        id_provider = IdProvider()
        deployment1.update_id_provider(id_provider)

        monostatic_sensors = [s.model_copy() for s in deployment1.monostatic_sensors]
        pcl_sensors = [s.model_copy() for s in deployment1.pcl_sensors]
        effectors = [e.model_copy() for e in deployment1.effectors]

        for sensor in deployment2.monostatic_sensors:
            sensor = sensor.model_copy()
            sensor.id += id_provider.increment(Entity.SENSOR)
            sensor.transmitter.id += id_provider.increment(Entity.TRANSMITTER)
            sensor.receiver.id += id_provider.increment(Entity.RECEIVER)
            monostatic_sensors.append(sensor)
        for sensor in deployment2.pcl_sensors:
            sensor = sensor.model_copy()
            sensor.id += id_provider.increment(Entity.SENSOR)
            sensor.transmitter.id += id_provider.increment(Entity.TRANSMITTER)
            sensor.receiver.id += id_provider.increment(Entity.RECEIVER)
            pcl_sensors.append(sensor)
        for effector in deployment2.effectors:
            effector = DirectFireEffector(
                id=effector.id + id_provider.increment(Entity.EFFECTOR),
                name=effector.name,
                point=effector.point.model_copy(),
                combat_range=effector.combat_range,
                n_attacks_left=effector.n_attacks_left,
                terrain=effector.terrain, # do not copy terrain!
            )
            effectors.append(effector)

        return StaticDispositive(
            monostatic_sensors=monostatic_sensors,
            pcl_sensors=pcl_sensors,
            effectors=effectors,
        )


class Dispositive(pydantic.BaseModel):
    static_dispositive: StaticDispositive
    mobile_dispositive: MobileDispositive

    def to_controller(
        self,
        terrain: AbstractTerrainModel,
        monostatic_sensor_rcs: float,
        id_provider: IdProvider,
        is_blue: bool,
    ) -> Controller:
        static_controller = self.static_dispositive.to_controller(
            monostatic_sensor_rcs,
            id_provider,
            is_blue,
        )
        mobile_controller = self.mobile_dispositive.to_controller(terrain)
        return ControllerGroup(controllers=[static_controller, mobile_controller])

    def update_id_provider(self, id_provider: IdProvider):
        self.static_dispositive.update_id_provider(id_provider)
        self.mobile_dispositive.update_id_provider(id_provider)


class PseudoTrackerParams(pydantic.BaseModel):
    removal_patience: int
    start_timestamp: float
    prior_position: Point


class TrackerFactory(pydantic.BaseModel):
    tracker_name: Literal["pseudotracker"] = "pseudotracker"
    parameters: PseudoTrackerParams

    def to_tracker(self, rng: np.random.Generator) -> AbstractTracker:
        if self.tracker_name == "pseudotracker":
            return PseudoTracker(
                removal_patience=self.parameters.removal_patience,
                rng=rng,
                start_time=datetime.datetime.fromtimestamp(
                    self.parameters.start_timestamp,
                    tz=datetime.UTC,
                ),
                prior_position=self.parameters.prior_position,
            )
        else:
            raise ValueError(f"Unknown tracker '{self.tracker_name}'")


class DamageModelFactory(pydantic.BaseModel):
    model_name: Literal["kill_always"]

    def to_damage_model(self, rng: np.random.Generator) -> AbstractDamageModel:
        if self.model_name == "kill_always":
            return UniformDamageModel(p_kill=1.0, rng=rng)
        else:
            raise RuntimeError(f"Unknown damage model {self.model_name}")


class ScenarioFactory(pydantic.BaseModel):
    name: str
    start_time: datetime.datetime
    """Start time of the scenario"""
    stop_time: datetime.datetime
    """Start time of the scenario"""
    time_step: int
    """Time step per iteration [s]"""
    blue_dispositive: Dispositive
    red_dispositive: Dispositive
    blue_tracker: TrackerFactory
    red_tracker: TrackerFactory
    terrain_model: TerrainFactory
    damage_model: DamageModelFactory
    seed: int

    def update_id_provider(self, id_provider: IdProvider):
        self.blue_dispositive.update_id_provider(id_provider)
        self.red_dispositive.update_id_provider(id_provider)

    def to_simulator(
        self,
        output_path: str,
        is_interactive: bool,
    ) -> tuple[Simulator, SituationalPictureBuffer | None]:
        id_provider = IdProvider()

        # Ensure consistent IDs.
        self.update_id_provider(id_provider)

        terrain = self.terrain_model.to_terrain()

        controller_blue = self.blue_dispositive.to_controller(
            terrain,
            1.0,
            id_provider,
            True,
        )
        controller_red = self.red_dispositive.to_controller(
            terrain,
            1.0,
            id_provider,
            False,
        )

        buffer = None
        listener = FileLogger(path=output_path)

        if is_interactive:
            buffer = SituationalPictureBuffer()
            listener = CompositeSimulationListener([listener, buffer])

        pcl_detector = PclDetector()
        pet_detector = PetDetector(terrain_model=terrain)

        rng = np.random.Generator(np.random.PCG64(seed=self.seed))

        min_time_step = datetime.timedelta(seconds=1 if is_interactive else 0)

        simulator = Simulator(
            pcl_detector=pcl_detector,
            pet_detector=pet_detector,
            visual_detector=None,
            blue_controller=controller_blue,
            red_controller=controller_red,
            blue_tracker=self.blue_tracker.to_tracker(rng),
            red_tracker=self.red_tracker.to_tracker(rng),
            start_time=self.start_time,
            time_step=datetime.timedelta(seconds=self.time_step),
            termination_criterion=TimeCriterion(end_time=self.stop_time),
            min_time_per_step=min_time_step,
            rng=rng,
            listener=listener,
            terrain_model=terrain,
            damage_model=self.damage_model.to_damage_model(rng),
            simulate_clutter=False,
            id_provider=id_provider,
            shot_association_tolerance=250.0,
        )

        return simulator, buffer
