from __future__ import annotations
import datetime
import json
from typing import Literal
import numpy as np
import pydantic

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
from theia.types import (
    AbstractTracker,
    ConstantRcsModel,
    Controller,
    Entity,
    IdProvider,
    MonostaticSensor,
    PetDetection,
    Point,
    Trajectory,
)


class TerrainFactory(pydantic.BaseModel):
    terrain_name: Literal["SRTM"]

    def to_terrain(self) -> AbstractTerrainModel:
        terrain = None
        if self.terrain_name == "SRTM":
            terrain = SrtmTerrainModel()
        else:
            raise ValueError(f"Unknown terrain model {self.terrain_name}")
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
    terrain: TerrainFactory

    def to_controller(self) -> Controller:
        return FixedPathOneWayDrone(
            effector=self.effector.to_effector(),
            trajectory=self.trajectory,
            assigned_goal=self.assigned_goal,
            terrain=self.terrain.to_terrain(),
        )


class MobileDispositive(pydantic.BaseModel):
    oneway_drones: list[FixedPathOneWayDroneFactory]

    def to_controller(self) -> Controller:
        oneway_drone_controllers = [
            drone.to_controller() for drone in self.oneway_drones
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
        return max([drone.trajectory.times[0] for drone in self.oneway_drones])

    @property
    def t_max(self) -> datetime.datetime:
        if len(self.oneway_drones) == 0:
            return None
        return max([drone.trajectory.times[-1] for drone in self.oneway_drones])


class StaticDispositive(pydantic.BaseModel):
    monostatic_sensors: list[MonostaticSensor]
    pcl_sensors: list[MonostaticSensor]
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
    def from_file(static_dispositive_file: str) -> tuple[StaticDispositive]:
        with open(static_dispositive_file, "r") as file:
            data = json.load(file)
            dispo = StaticDispositive.model_validate(data)

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


class Dispositive(pydantic.BaseModel):
    static_dispositive: StaticDispositive
    mobile_dispositive: MobileDispositive

    def to_controller(
        self,
        monostatic_sensor_rcs: float,
        id_provider: IdProvider,
        is_blue: bool,
    ) -> Controller:
        static_controller = self.static_dispositive.to_controller(
            monostatic_sensor_rcs,
            id_provider,
            is_blue,
        )
        mobile_controller = self.mobile_dispositive.to_controller()
        return ControllerGroup(controllers=[static_controller, mobile_controller])

    def update_id_provider(self, id_provider: IdProvider):
        self.static_dispositive.update_id_provider(id_provider)
        self.mobile_dispositive.update_id_provider(id_provider)


class PseudoTrackerParams(pydantic.BaseModel):
    removal_patience: int
    start_timestamp: int
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
    start_time: int
    """Start time of the scenario (UNIX epoch)"""
    stop_time: int
    """Start time of the scenario (UNIX epoch)"""
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

        controller_blue = self.blue_dispositive.to_controller(
            1.0,
            id_provider,
            True,
        )
        controller_red = self.red_dispositive.to_controller(
            1.0,
            id_provider,
            False,
        )

        buffer = None
        listener = FileLogger(path=output_path)

        if is_interactive:
            buffer = SituationalPictureBuffer()
            listener = CompositeSimulationListener([listener, buffer])

        terrain = self.terrain_model.to_terrain()

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
            start_time=datetime.datetime.fromtimestamp(
                self.start_time,
                tz=datetime.UTC,
            ),
            time_step=datetime.timedelta(seconds=self.time_step),
            termination_criterion=TimeCriterion(
                end_time=datetime.datetime.fromtimestamp(
                    self.stop_time, tz=datetime.UTC
                )
            ),
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
