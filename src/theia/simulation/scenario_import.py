from __future__ import annotations
import json
from typing import Literal
import pydantic

from theia.effectors import DirectFireEffector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.fixed_path_kamikaze_drone import FixedPathOneWayDrone
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.terrain import AbstractTerrainModel, SrtmTerrainModel
from theia.types import (
    ConstantRcsModel,
    Controller,
    Entity,
    IdProvider,
    MonostaticSensor,
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


class TrajectoryCollection(pydantic.BaseModel):
    oneway_trajectories: list[Trajectory]

    def to_controller(self) -> Controller:
        controllers = [FixedPathOneWayDrone(t) for t in self.trajectories]
        return ControllerGroup(controllers=controllers)


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


class ScenarioFactory(pydantic.BaseModel):
    name: str
    blue_dispositive: Dispositive
    red_dispositive: Dispositive
