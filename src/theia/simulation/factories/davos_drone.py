import datetime
import json

import numpy as np

from theia.config import SIDC
from theia.coordinates import CoordinateTransformations
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.effectors import DirectFireEffector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.fixed_path_kamikaze_drone import FixedPathOneWayDrone
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.static_gbad_controller import StaticGbadController
from theia.simulation.controllers.static_gbad_coordinator import StaticGbadCoordinator
from theia.simulation.controllers.stationary_eye_controller import (
    StationaryEyeController,
)
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.factories.abstract_simulator_factory import (
    AbstractSimulatorFactory,
)
from theia.simulation.simulator import TerminationCriterion, TimeCriterion
from theia.simulation.trackers.pseudo_tracker import PseudoTracker
from theia.terrain import AbstractTerrainModel
from theia.test_data import build_flores_monostatic_radar
from theia.types import (
    AbstractTracker,
    ConstantRcsModel,
    Controller,
    Entity,
    IdProvider,
    Point,
    Receiver,
    Trajectory,
    VisualSensor,
)


class DavosDroneScenarioFactory(AbstractSimulatorFactory):
    def __init__(self, rng: np.random.Generator, terrain_model: AbstractTerrainModel):
        self._rng = rng
        self._terrain_model = terrain_model
        self._t0 = datetime.datetime(
            year=2026,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
        )
        self._tmax = self._t0 + datetime.timedelta(hours=1)

        self.BLUE_INFRA_LAT = 46.80089
        self.BLUE_INFRA_LON = 9.83124
        self._p_infra = Point(
            lat=self.BLUE_INFRA_LAT,
            lon=self.BLUE_INFRA_LON,
            alt=self._terrain_model.elevationAt(
                self.BLUE_INFRA_LAT, self.BLUE_INFRA_LON
            )
            + 10,
        )

        # Load trajectory.
        data_dir = "/home/user/Documents/theia_backend/examples/conference/data"
        with open(f"{data_dir}/flight-path-spacetime_adversarial.json", "r") as file:
            data = json.load(file)
            self._times = [
                self._t0 + datetime.timedelta(seconds=p["time"] - 1550)
                for p in data
                if p["time"] > 1550
            ]
            self._points = [
                Point(lat=p["lat"], lon=p["lon"], alt=p["alt"])
                for p in data
                if p["time"] > 1550
            ]
        self._id_provider = IdProvider()

    def _build_single_drone(
        self,
        spatial_offset: tuple[float, float, float],
        temporal_offset: datetime.timedelta,
        name: str = "",
    ) -> Controller:
        points_ecef = [
            np.array(
                CoordinateTransformations.geodetic_to_cartesian(p.lat, p.lon, p.alt)
            )
            + np.array(spatial_offset)
            for p in self._points
        ]
        points: list[Point] = [
            CoordinateTransformations.cartesian_to_geodetic(*p) for p in points_ecef
        ]
        # Make sure that all the drones aim at the conference center, regardless
        # of their trajectory.
        points[-1] = (
            self._points[-1].lat,
            self._points[-1].lon,
            self._points[-1].alt,
        )

        trajectory = Trajectory(
            target_id=self._id_provider.increment(Entity.TARGET, name),
            target_sidc=SIDC.RED_FIXED_WING,
            times=[t + temporal_offset for t in self._times],
            lats=[p[0] for p in points],
            lons=[p[1] for p in points],
            alts=[p[2] for p in points],
            vxs=[0.0 for _ in range(len(points))],
            vys=[0.0 for _ in range(len(points))],
            vzs=[0.0 for _ in range(len(points))],
            cross_section_model=ConstantRcsModel(rcs=1.0),
        )
        effector = DirectFireEffector(
            id=self._id_provider.increment(Entity.EFFECTOR),
            name=name,
            point=self._p_infra,
            combat_range=200.0,
            n_attacks_left=1,
            terrain=self._terrain_model,
            # One-shot drone - no rate-of-fire limit applies.
            cadence=float("inf"),
        )
        controller = LivingController(
            child=FixedPathOneWayDrone(
                effector=effector,
                trajectory=trajectory,
                assigned_goal=self._p_infra,
                terrain=self._terrain_model,
            ),
            target_id=trajectory.target_id,
        )
        return controller

    def _build_drone_flock(
        self,
        N: int,
        temporal_offset: datetime.timedelta = datetime.timedelta(),
        flock_name: str = "",
    ):
        controllers = []
        for i in range(N):
            offset = self._rng.uniform(100.0, 200.0, size=(3,))
            controllers.append(
                self._build_single_drone(
                    offset, temporal_offset, name=f"{flock_name}.{i + 1}"
                )
            )
        return ControllerGroup(controllers=controllers)

    def _build_conference_center_observer(self) -> Controller:
        """Incorporate the prior knowledge that the position of the conference center is known."""
        conf_center_observer = StationaryEyeController(
            sensor=VisualSensor(
                id=self._id_provider.increment(Entity.SENSOR),
                receiver=Receiver(
                    id=0,
                    point=self._p_infra,
                    antenna_height=0.0,
                    diameter=0.0,
                    cpi_pulses=0,
                    pfa=0.0,
                    min_elevation=0.0,
                    max_elevation=np.pi / 2.0,
                    min_azimuth=0.0,
                    max_azimuth=2 * np.pi,
                    rotation_time=0,
                    bandwidth=0,
                ),
                detection_range=10.0,
            ),
        )
        return conf_center_observer

    def _build_conference_center_controller(self) -> Controller:
        # Define critical infrastructure.
        target_id = self._id_provider.increment(Entity.TARGET, "Conference center")

        return LivingController(
            child=WaypointTargetController(
                name="Conference center",
                sidc=SIDC.BLUE_GOVERNMENT_SITE,
                target_id=target_id,
                times=[self._t0, self._tmax],
                waypoints=[self._p_infra, self._p_infra],
                rcs_model=ConstantRcsModel(rcs=100),
            ),
            target_id=target_id,
        )

    def _build_oerlikon_gdf(
        self,
        point: Point,
        geojson_alts: list[float] = [2000.0],
        name: str = "",
    ) -> Controller:
        # Source:
        # https://en.wikipedia.org/wiki/Oerlikon_GDF (2026-07-14)
        # https://weaponsystems.net/system/933-EE02%20-%20GDF (2026-07-14)
        # According to Wikipedia, there are 280 rounds in total, and a typical
        # engagement burst consists of 28 rounds.
        # Therefore, there are approx. 280 / 28 = 10 attacks.
        effector = DirectFireEffector(
            id=self._id_provider.increment(Entity.EFFECTOR, f"Flak @ {point}"),
            name=name,
            point=point,
            combat_range=4_000,
            n_attacks_left=10,
            terrain=self._terrain_model,
            # One burst per second - placeholder, no cited source.
            cadence=1.0,
        )

        return StaticGbadController(
            target_id=self._id_provider.increment(Entity.TARGET, f"Flak @ {point}"),
            sidc=SIDC.BLUE_AIR_DEFENCE,
            rcs=2.0,
            effector=effector,
            assigned_track_id=None,
            geojson_range_altitudes=geojson_alts,
        )

    def _get_pcl_detector(self) -> PclDetector:
        return PclDetector()

    def _get_pet_detector(self) -> PetDetector:
        return PetDetector(terrain_model=self._terrain_model)

    def _get_blue_tracker(self) -> AbstractTracker:
        return PseudoTracker(
            removal_patience=30,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_red_tracker(self) -> AbstractTracker:
        return PseudoTracker(
            removal_patience=0,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_blue_controller(self) -> Controller:
        conf_center_controller = self._build_conference_center_controller()
        p = Point(
            lat=46.82579,
            lon=9.85658,
            alt=self._terrain_model.elevationAt(46.82579, 9.85658),
        )
        p_flak_klosters = Point(
            lat=46.87991,
            lon=9.87260,
            alt=self._terrain_model.elevationAt(46.87991, 9.87260) + 10.0,
        )
        flak = self._build_oerlikon_gdf(p, name="Flak Davos")
        flak_klosters = self._build_oerlikon_gdf(p_flak_klosters, name="Flak Klosters")
        p_klosters = Point(
            lat=46.86938,
            lon=9.88244,
            alt=self._terrain_model.elevationAt(46.86938, 9.88244),
        )
        radar_flak = build_flores_monostatic_radar(
            p,
            sensor_id=self._id_provider.increment(Entity.SENSOR),
            rx_id=0,
            tx_id=0,
        )
        radar_klosters = build_flores_monostatic_radar(
            p_klosters,
            sensor_id=self._id_provider.increment(Entity.SENSOR),
            rx_id=1,
            tx_id=1,
        )
        radar_flak_controller = MonostaticRadarController(
            target_id=self._id_provider.increment(Entity.TARGET),
            radar=radar_flak,
            is_blue=True,
            rcs_model=ConstantRcsModel(rcs=1.0),
            name="Flak Radar",
        )
        radar_klosters_controller = MonostaticRadarController(
            target_id=self._id_provider.increment(Entity.TARGET),
            radar=radar_klosters,
            is_blue=True,
            rcs_model=ConstantRcsModel(rcs=1.0),
            name="Klosters Radar",
        )
        return ControllerGroup(
            [
                conf_center_controller,
                StaticGbadCoordinator(
                    controllers=[flak, flak_klosters],
                    terrain=self._terrain_model,
                ),
                radar_flak_controller,
                radar_klosters_controller,
            ]
        )

    def _get_red_controller(self) -> Controller:
        conf_center_observer = self._build_conference_center_observer()
        # controller = self._build_single_drone(
        #     spatial_offset=(0, 0, 0), temporal_offset=datetime.timedelta(seconds=0)
        # )
        controllers = []
        for i in range(5):
            controllers.append(
                self._build_drone_flock(
                    5,
                    temporal_offset=datetime.timedelta(minutes=i * 2),
                    flock_name=f"Drone flock {i + 1}",
                )
            )
        drone_controller = ControllerGroup(controllers=controllers)
        print(dict(sorted(self._id_provider._names.items())))
        return ControllerGroup(controllers=[drone_controller, conf_center_observer])

    def _get_start_time(self) -> datetime.datetime:
        return self._t0

    def _get_timestep(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=1)

    def _get_termination_criterion(self) -> TerminationCriterion:
        return TimeCriterion(self._tmax)
