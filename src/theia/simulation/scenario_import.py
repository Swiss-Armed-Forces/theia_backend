from __future__ import annotations

import datetime
import json
from typing import Literal, Optional

import matplotlib
import numpy as np
import pydantic

from theia.config import SIDC, TERRAIN_HBV_DATA_DIR
from theia.coordinates import CoordinateTransformations
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.distance import line_of_sight_distance
from theia.effectors import DirectFireEffector
from theia.grids import LatLonHeightGrid
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.fixed_path_kamikaze_drone import FixedPathOneWayDrone
from theia.simulation.controllers.geojson_controller import GeoJsonController
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.static_direct_fire_controller import (
    StaticDirectFireController,
)
from theia.simulation.controllers.static_direct_fire_coordinator import (
    StaticDirectFireCoordinator,
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
    GeoJSONFeature,
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
            terrain = FastSrtmModel(
                tree=tree,
                srtm_model=SrtmTerrainModel(),
                t_min=30.0,
            )
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

    def to_effector(self, terrain: AbstractTerrainModel) -> DirectFireEffector:
        return DirectFireEffector(
            id=self.id,
            name=self.name,
            point=self.point,
            combat_range=self.combat_range,
            n_attacks_left=self.n_attacks_left,
            terrain=terrain,
        )


class StaticDirectFireEffectorFactory(pydantic.BaseModel):
    target_id: int
    rcs: float
    effector: DirectFireEffectorFactory

    def to_controller(
        self,
        terrain: AbstractTerrainModel,
        is_blue: bool,
    ) -> StaticDirectFireController:
        c = StaticDirectFireController(
            target_id=self.target_id,
            sidc=SIDC.BLUE_AIR_DEFENCE if is_blue else SIDC.RED_AIR_DEFENCE,
            rcs=self.rcs,
            effector=self.effector.to_effector(terrain),
        )
        return c


class FixedPathOneWayDroneFactory(pydantic.BaseModel):
    effector: DirectFireEffectorFactory
    trajectory: Trajectory
    assigned_goal: Point

    def to_controller(self, terrain: AbstractTerrainModel) -> Controller:
        drone = FixedPathOneWayDrone(
            effector=self.effector.to_effector(terrain),
            trajectory=self.trajectory,
            assigned_goal=self.assigned_goal,
            terrain=terrain,
        )
        return LivingController(child=drone, target_id=drone.trajectory.target_id)


class MonostaticSensorFactory(pydantic.BaseModel):
    target_id: int
    rcs: float
    sensor: MonostaticSensor

    def to_controller(self, is_blue: bool) -> MonostaticRadarController:
        c = MonostaticRadarController(
            target_id=self.target_id,
            radar=self.sensor,
            is_blue=is_blue,
            rcs_model=ConstantRcsModel(rcs=self.rcs),
        )
        return LivingController(child=c, target_id=self.target_id)


class MonostaticCoverageCalcSettings(pydantic.BaseModel):
    targetAlt: float
    targetRcs: float
    probabilityThreshold: float
    azimuthResolution: float
    rangeOnly: bool


class MonostaticCoverageCalculation(pydantic.BaseModel):
    sensorId: int
    settings: MonostaticCoverageCalcSettings
    coverage: GeoJSONFeature


class PclSensorFactory(pydantic.BaseModel):
    target_id: int
    rcs: float
    sensor: PclSensor

    # TODO: Build a controller from this!


class PclMinDetectableRcsCalcSettings(pydantic.BaseModel):
    grid: LatLonHeightGrid
    snr_threshold: float
    """SNR theshold [dB]"""
    doppler_threshold: float
    """Doppler theshold [Hz]"""
    delay_threshold: float
    """Delay theshold [us]"""


class PclMinDetectableRcsCalculation(pydantic.BaseModel):
    sensor_id: int
    settings: PclMinDetectableRcsCalcSettings
    values: list[list[list[float]]]


class SimulationResults(pydantic.BaseModel):
    monostaticCoverages: list[tuple[MonostaticCoverageCalculation, datetime.datetime]]
    """calculation result, date of calculation [epochs]"""
    pclMinDetectableRcsGrids: list[
        tuple[PclMinDetectableRcsCalculation, datetime.datetime]
    ]
    """calculation result, date of calculation [epochs]"""

    def to_geojson_dict(self) -> dict[str, GeoJSONFeature]:
        result: dict[str, GeoJSONFeature] = {}
        for calc, _ in self.monostaticCoverages:
            tag = f"Monostatic #{calc.sensorId}, {calc.settings.targetAlt:.0f} MASL, RCS={calc.settings.targetRcs:.1f}m^2"
            result[tag] = calc.coverage
        return result


class OrderOfBattle(pydantic.BaseModel):
    monostatic_sensors: list[MonostaticSensorFactory]
    pcl_sensors: list[PclSensorFactory]
    effectors: list[StaticDirectFireEffectorFactory]
    simulationResults: SimulationResults
    oneway_drones: list[FixedPathOneWayDroneFactory]
    ballistic_missiles: list[BallisticMissileFactory]
    unused_id_sensor: int
    unused_id_receiver: int
    unused_id_transmitter: int
    unused_id_effector: int
    unused_target_id: int

    @staticmethod
    def from_file(orbat_file: str) -> OrderOfBattle:
        with open(orbat_file, "r") as file:
            data = json.load(file)
            orbat = OrderOfBattle.model_validate(data)
        # Validate IDs.
        id_provider = IdProvider()
        orbat.update_id_provider(id_provider)
        consistent_ids = (
            orbat.unused_id_sensor >= id_provider._free_ids[Entity.SENSOR]
            and orbat.unused_id_receiver >= id_provider._free_ids[Entity.RECEIVER]
            and orbat.unused_id_transmitter >= id_provider._free_ids[Entity.TRANSMITTER]
            and orbat.unused_id_effector >= id_provider._free_ids[Entity.EFFECTOR]
            and orbat.unused_target_id >= id_provider._free_ids[Entity.TARGET]
        )
        if not consistent_ids:
            raise ValueError("ORBAT file is inconsistent: Unused IDs are used")
        return orbat

    def to_controller(
        self,
        terrain: AbstractTerrainModel,
        is_blue: bool,
    ) -> Controller:
        monostatic_controllers = [
            s.to_controller(is_blue) for s in self.monostatic_sensors
        ]

        ##########################
        # TODO: Load PCL sensors.
        ##########################

        static_deployment_controllers = [
            LivingController(
                child=StaticDirectFireCoordinator(
                    controllers=[e.to_controller(terrain, is_blue)],
                    terrain=terrain,
                ),
                target_id=e.target_id,
            )
            for e in self.effectors
        ]

        oneway_drone_controllers = [
            drone.to_controller(terrain) for drone in self.oneway_drones
        ]
        bm_controllers = [bm.to_controller(is_blue) for bm in self.ballistic_missiles]

        return GeoJsonController(
            child=ControllerGroup(
                controllers=monostatic_controllers
                + static_deployment_controllers
                + oneway_drone_controllers
                + bm_controllers
            ),
            geojson_features=self.simulationResults.to_geojson_dict(),
        )

    def update_id_provider(self, id_provider: IdProvider):
        for detectable_sensor in self.monostatic_sensors:
            sensor = detectable_sensor.sensor
            id_provider.register_entity(Entity.SENSOR, sensor.id)
            id_provider.register_entity(Entity.TRANSMITTER, sensor.transmitter.id)
            id_provider.register_entity(Entity.RECEIVER, sensor.receiver.id)
            id_provider.register_entity(Entity.TARGET, detectable_sensor.target_id)
        for detectable_sensor in self.pcl_sensors:
            sensor = detectable_sensor.sensor
            if sensor.receiver.id in id_provider._used_ids[Entity.RECEIVER]:
                raise ValueError(f"PCL Receiver ID duplicated: {sensor.receiver.id}")
            if sensor.transmitter.id in id_provider._used_ids[Entity.TRANSMITTER]:
                raise ValueError(
                    f"PCL Transmitter ID duplicated: {sensor.transmitter.id}"
                )
        for detectable_sensor in self.pcl_sensors:
            sensor = detectable_sensor.sensor
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
        for detectable_effector in self.effectors:
            effector = detectable_effector.effector
            id_provider.register_entity(Entity.EFFECTOR, effector.id)
            id_provider.register_entity(Entity.TARGET, detectable_effector.target_id)
        for drone in self.oneway_drones:
            id_provider.register_entity(Entity.EFFECTOR, drone.effector.id)
            id_provider.register_entity(Entity.TARGET, drone.trajectory.target_id)
        for missile in self.ballistic_missiles:
            id_provider.register_entity(Entity.EFFECTOR, missile.effector_id)
            id_provider.register_entity(Entity.TARGET, missile.target_id)

    def reindex(self, id_provider: IdProvider):
        sensor_id_mapping: dict[int, int] = {}
        for detectable_sensor in self.monostatic_sensors:
            s = detectable_sensor.sensor
            sensor_id_mapping[s.id] = id_provider.increment(Entity.SENSOR)
            s.id = sensor_id_mapping[s.id]
            s.transmitter.id = id_provider.increment(Entity.TRANSMITTER)
            s.receiver.id = id_provider.increment(Entity.RECEIVER)
        pcl_receiver_id_mapping: dict[int, int] = {}
        for detectable_sensor in self.pcl_sensors:
            s = detectable_sensor.sensor
            sensor_id_mapping[s.id] = id_provider.increment(Entity.SENSOR)
            s.id = sensor_id_mapping[s.id]
            s.transmitter.id = id_provider.increment(Entity.TRANSMITTER)

            # Make sure that the receiver ID is only updated once and kept consistent.
            if s.receiver.id in pcl_receiver_id_mapping:
                s.receiver.id = pcl_receiver_id_mapping[s.receiver.id]
            else:
                new_id = id_provider.increment(Entity.RECEIVER)
                pcl_receiver_id_mapping[s.receiver.id] = new_id
                s.receiver.id = new_id
        for detectable_effector in self.effectors:
            detectable_effector.effector.id = id_provider.increment(Entity.EFFECTOR)
        for calc, t in self.simulationResults.monostaticCoverages:
            calc.sensorId = sensor_id_mapping[calc.sensorId]
        for calc, t in self.simulationResults.pclMinDetectableRcsGrids:
            calc.sensor_id = sensor_id_mapping[calc.sensor_id]
        for drone in self.oneway_drones:
            drone.effector.id = id_provider.increment(Entity.EFFECTOR)
            new_id = id_provider.increment(Entity.TARGET)
            print(
                f"Update trajectory targe_id {drone.trajectory.target_id} -> {new_id}"
            )
            drone.trajectory.target_id = new_id
        for missile in self.ballistic_missiles:
            missile.effector_id = id_provider.increment(Entity.EFFECTOR)
            missile.target_id = id_provider.increment(Entity.TARGET)

    @property
    def t_min(self) -> datetime.datetime | None:
        if len(self.oneway_drones) == 0 and len(self.ballistic_missiles) == 0:
            return None
        return min(
            [drone.trajectory.times[0] for drone in self.oneway_drones]
            + [bm.get_trajectory(True).times[0] for bm in self.ballistic_missiles]
        )

    @property
    def t_max(self) -> datetime.datetime:
        if len(self.oneway_drones) == 0 and len(self.ballistic_missiles) == 0:
            return None
        return max(
            [drone.trajectory.times[-1] for drone in self.oneway_drones]
            + [bm.get_trajectory(True).times[-1] for bm in self.ballistic_missiles]
        )

    @staticmethod
    def merge(
        orbat1: OrderOfBattle,
        orbat2: OrderOfBattle,
    ) -> OrderOfBattle:
        id_provider = IdProvider()
        orbat1.update_id_provider(id_provider)
        orbat2.reindex(id_provider)

        return OrderOfBattle(
            monostatic_sensors=orbat1.monostatic_sensors + orbat2.monostatic_sensors,
            pcl_sensors=orbat1 + orbat2.pcl_sensors,
            effectors=orbat1.effectors + orbat2.effectors,
            simulationResults=orbat1.simulationResults + orbat2.simulationResults,
            oneway_drones=orbat1.oneway_drones + orbat2.oneway_drones,
            ballistic_missiles=orbat1.ballistic_missiles + orbat2.oneway_drones,
            unused_id_sensor=id_provider.increment(Entity.SENSOR),
            unused_id_receiver=id_provider.increment(Entity.RECEIVER),
            unused_id_transmitter=id_provider.increment(Entity.TRANSMITTER),
            unused_id_effector=id_provider.increment(Entity.EFFECTOR),
            unused_target_id=id_provider.increment(Entity.TARGET),
        )

    @staticmethod
    def to_openapi_schema() -> dict:
        schema = OrderOfBattle.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        defs = schema.pop("$defs", {})
        return {
            "openapi": "3.1.0",
            "info": {"title": "Theia", "version": "0.0.0"},
            "paths": {},
            "components": {
                "schemas": {
                    "OrderOfBattle": schema,
                    **defs,
                },
            },
        }


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
    blue_orbat: OrderOfBattle
    red_orbat: OrderOfBattle
    blue_tracker: TrackerFactory
    red_tracker: TrackerFactory
    terrain_model: TerrainFactory
    damage_model: DamageModelFactory
    seed: int
    blue_geojson: dict[str, list[GeoJSONFeature]] = {}
    """List of GeoJSON objects for BLUE that belong to the same tag."""
    red_geojson: dict[str, list[GeoJSONFeature]] = {}
    """List of GeoJSON objects for RED that belong to the same tag."""

    def update_id_provider(self, id_provider: IdProvider):
        self.blue_orbat.update_id_provider(id_provider)
        self.red_orbat.update_id_provider(id_provider)

    def to_simulator(
        self,
        output_path: str,
        is_interactive: bool,
    ) -> tuple[Simulator, SituationalPictureBuffer | None]:
        id_provider = IdProvider()

        # Ensure consistent IDs.
        self.update_id_provider(id_provider)

        terrain = self.terrain_model.to_terrain()

        controller_blue = self.blue_orbat.to_controller(terrain, True)
        controller_red = self.red_orbat.to_controller(terrain, False)

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


class BallisticMissileFactory(pydantic.BaseModel):
    p_start: Point
    p_stop: Point
    t_start: datetime.datetime
    terrain: TerrainFactory
    target_id: int
    effector_id: int
    rcs: float
    alpha: Optional[float] = 45.0
    """Launch angle w. r. t. LOS [°]"""

    def model_post_init(self, __context):
        self._v0, self._times, self._trajectory_data = _build_trajectory(
            np.deg2rad(self.alpha),
            self.p_start,
            self.p_stop,
        )

    def to_controller(self, is_blue: bool) -> FixedPathOneWayDrone:
        bm = FixedPathOneWayDrone(
            effector=DirectFireEffector(
                id=self.effector_id,
                name="ballistic missile",
                point=self.p_start,
                combat_range=100.0,
                n_attacks_left=1,
                terrain=self.terrain.to_terrain(),
            ),
            trajectory=self.get_trajectory(is_blue),
            assigned_goal=self.p_stop,
            terrain=self.terrain.to_terrain(),
        )
        return LivingController(child=bm, target_id=bm.trajectory.target_id)

    def get_trajectory(self, is_blue: bool) -> Trajectory:
        points = _convert_to_geodetic(self.p_start, self.p_stop, self._trajectory_data)
        return Trajectory(
            target_id=self.target_id,
            target_sidc=SIDC.BLUE_MISSILE if is_blue else SIDC.RED_MISSILE,
            times=[
                datetime.datetime.fromtimestamp(t, tz=datetime.UTC) for t in self._times
            ],
            lats=[p.lat for p in points],
            lons=[p.lon for p in points],
            alts=[p.alt for p in points],
            vxs=[0.0 for p in points],
            vys=[0.0 for p in points],
            vzs=[0.0 for p in points],
            cross_section_model=ConstantRcsModel(rcs=self.rcs),
        )

    def plot_trajectory(self) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        fig, ax = matplotlib.pyplot.subplots()
        data = _build_earth_curvature(self.p_start, self.p_stop) / 1e3
        ax.plot(
            data[:, 0],
            data[:, 1],
            "-",
            label="earth surface at sea level",
            color="black",
        )
        data = _build_earth_curvature(self.p_start, self.p_stop, 100_000) / 1e3
        ax.plot(
            data[:, 0],
            data[:, 1],
            "--",
            label="Karman line ('space')",
            color="black",
        )

        ax.plot(
            self._trajectory_data[:, 0] / 1e3,
            self._trajectory_data[:, 1] / 1e3,
            "-",
            label=rf"Trajectory for $\alpha = {self.alpha} \degree$; T = {self._times[-1] / 60:.1f}min; $v_0$ = {self._v0:.0f} m/s",
        )

        ax.set_xlabel("LOS distance [km]", fontsize=14)
        ax.set_ylabel("Altitude above LOS [km]", fontsize=14)
        ax.legend()
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        return fig, ax


def _build_trajectory(alpha: float, p_start: Point, p_stop: Point, dt: int = 1):
    """
    Calculate a ballistic trajectory that connects the start and stop point.

    The only force considered is gravity.

    Parameters
    ----------
    alpha: float
        Start angle of the ballistic motion [rad] in the 2D coordinate system
        (see "Returns")
    p_start: Point
        Start point of the trajectory
    p_stop: Point
        Stop point of the trajectory
    dt: int
        Number of seconds to use for representing the trajectory

    Returns
    -------
    v0: float
        Initial velocity [m/s]
    times: np.ndarray
        Time steps of the trajectory [s]. Shape (N,).
    trajectory: np.ndarray
        Points along the trajectory at 10s resolution. Shape (N, 2).
        Column 0 contains the horizontal component, column 1 the vertical component
        of the trajectory.
        The coordinate system is spanned by the LOS from start to end point and
        the parto of radial direction at the start point that is orthogonal to the LOS.
    """
    distance = line_of_sight_distance(*p_start.as_tuple(), *p_stop.as_tuple())

    p_start_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_start.as_tuple())
    )
    p_stop_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_stop.as_tuple())
    )
    direction = p_stop_ecef - p_start_ecef
    direction /= np.linalg.norm(direction)

    v0 = np.sqrt(9.81 * distance / (2 * np.cos(alpha) * np.sin(alpha)))
    t_stop = 2 * np.sin(alpha) * v0 / 9.81

    times = np.arange(0, t_stop, dt).tolist() + [t_stop]
    xs = []
    ys = []
    for t in times:
        x = np.cos(alpha) * v0 * t
        y = np.sin(alpha) * v0 * t - 0.5 * 9.81 * t**2
        xs.append(x)
        ys.append(y)

    if t_stop % 1 != 0:
        times.append(np.ceil(t_stop))
        xs.append(xs[-1])
        ys.append(ys[-1])

    return v0, times, np.vstack([xs, ys]).T


def _build_earth_curvature(p_start, p_stop, altitude=0):
    distance = line_of_sight_distance(*p_start.as_tuple(), *p_stop.as_tuple())

    p_start_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_start.as_tuple())
    )
    p_stop_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_stop.as_tuple())
    )
    direction = p_stop_ecef - p_start_ecef
    direction /= np.linalg.norm(direction)

    xs = list(np.arange(0, distance, 100))
    ys = []
    for d in xs:
        p_los_ecef = p_start_ecef + d * direction
        p_los_geodetic = CoordinateTransformations.cartesian_to_geodetic(*p_los_ecef)
        y = altitude - p_los_geodetic[2]
        ys.append(y)

    return np.vstack([xs, ys]).T


def _convert_to_geodetic(
    p_start: Point,
    p_stop: Point,
    trajectory: np.ndarray,
) -> list[Point]:
    p_start_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_start.as_tuple())
    )
    p_stop_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_stop.as_tuple())
    )

    x_direction = p_stop_ecef - p_start_ecef
    x_direction /= np.linalg.norm(x_direction)
    y_direction = p_start_ecef / np.linalg.norm(p_start_ecef)
    y_direction -= np.dot(x_direction, y_direction) * x_direction
    assert np.isclose(np.dot(x_direction, y_direction), 0)

    points: list[Point] = []
    for x, y in trajectory:
        p_ecef = p_start_ecef + x * x_direction + y * y_direction
        p = CoordinateTransformations.cartesian_to_geodetic(*p_ecef)
        points.append(Point(lat=p[0], lon=p[1], alt=p[2]))
    return points
