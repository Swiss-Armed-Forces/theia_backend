import datetime
import json

import numpy as np

from theia.config import SIDC
from theia.coordinates import CoordinateTransformations
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.distance import burstvincentydistance
from theia.effectors import DirectFireEffector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.fixed_path_kamikaze_drone import FixedPathOneWayDrone
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


class ZurichAirportScenarioFactory(AbstractSimulatorFactory):
    def __init__(
        self,
        rng: np.random.Generator,
        terrain_model: AbstractTerrainModel,
        N_attackers: int = 200,
    ):
        self._rng = rng
        self._terrain_model = terrain_model
        self._N_attackers = N_attackers
        self._t0 = datetime.datetime(
            year=2026,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            tzinfo=datetime.timezone.utc,
        )
        self._tmax = self._t0 + datetime.timedelta(hours=1)

        self.BLUE_INFRA_LAT = 47.45270
        self.BLUE_INFRA_LON = 8.56068
        self._p_infra = Point(
            lat=self.BLUE_INFRA_LAT,
            lon=self.BLUE_INFRA_LON,
            alt=self._terrain_model.elevationAt(
                self.BLUE_INFRA_LAT, self.BLUE_INFRA_LON
            )
            + 10,
        )

        # Load trajectory.
        # TODO

        self._id_provider = IdProvider()

    def _build_single_drone(
        self,
        bearing: float,  # deg
        t_start: datetime.datetime,
        alt: float = 1000.0,
        d_start: float = 150_000.0,
        d_descent: float = 10_000.0,
        speed: float = 300,  # m / s
        name: str = "",
    ) -> Controller:
        points: list[Point] = []
        times = []
        # Start at d_start from the airport and move toward the airport.
        # Travel at altitude alt until you reach d_descent, then descend linearly
        # to the airport.
        for i, d in enumerate(np.arange(0.0, d_start, step=300.0)[::-1]):
            p = burstvincentydistance(
                (self._p_infra.lat, self._p_infra.lon),
                d,
                bearing,
                alt
                if d > d_descent
                else self._p_infra.alt + d * (alt - self._p_infra.alt) / d_descent,
            )
            points.append(p)
            times.append(t_start + datetime.timedelta(seconds=i))
        # The drones that arrive at the target point just stand still.
        if times[-1] < self._tmax:
            points.append(points[-1])
            times.append(self._tmax)

        trajectory = Trajectory(
            target_id=self._id_provider.increment(Entity.TARGET, name),
            target_sidc=SIDC.RED_FIXED_WING,
            times=times,
            lats=[p.lat for p in points],
            lons=[p.lon for p in points],
            alts=[p.alt for p in points],
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

    def _build_airport_observer(self) -> Controller:
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

    def _build_airport_controller(self) -> Controller:
        # Define critical infrastructure.
        target_id = self._id_provider.increment(Entity.TARGET, "Airport Zurich")

        return LivingController(
            child=WaypointTargetController(
                name="Airport Zürich",
                sidc=SIDC.BLUE_AIRPORT,
                target_id=target_id,
                times=[self._t0, self._tmax],
                waypoints=[self._p_infra, self._p_infra],
                rcs_model=ConstantRcsModel(rcs=100),
            ),
            target_id=target_id,
        )

    def _build_bodluv(
        self,
        point: Point,
        combat_range: float = 10_000,
        n_attacks: int = 3,
        geojson_alts: list[float] = [2000.0],
        name: str = "",
    ) -> Controller:
        # Source for default values:
        # https://en.wikipedia.org/wiki/Oerlikon_GDF (2026-07-14)
        # https://weaponsystems.net/system/933-EE02%20-%20GDF (2026-07-14)
        # According to Wikipedia, there are 280 rounds in total, and a typical
        # engagement burst consists of 28 rounds.
        # Therefore, there are approx. 280 / 28 = 10 attacks.
        effector = DirectFireEffector(
            id=self._id_provider.increment(Entity.EFFECTOR, name),
            name=name,
            point=point,
            combat_range=combat_range,
            n_attacks_left=n_attacks,
            terrain=self._terrain_model,
        )

        return StaticDirectFireController(
            target_id=self._id_provider.increment(Entity.TARGET, name),
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
            removal_patience=2,
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
        conf_center_controller = self._build_airport_controller()
        # BLUE sensors.
        radar_airport = build_flores_monostatic_radar(
            point=Point(
                lat=47.46274,
                lon=8.56923,
                alt=self._terrain_model.elevationAt(47.46274, 8.56923),
            ),
            sensor_id=self._id_provider.increment(Entity.SENSOR, name="Airport radar"),
            rx_id=0,
            tx_id=0,
        )
        radar_airport.receiver.rotation_time = 2
        radar_schoneberg = build_flores_monostatic_radar(
            point=Point(
                lat=47.63549,
                lon=8.88992,
                alt=self._terrain_model.elevationAt(47.63549, 8.88992),
            ),
            sensor_id=self._id_provider.increment(Entity.SENSOR, name="Schönebärg"),
            rx_id=1,
            tx_id=1,
        )
        radar_schoneberg.receiver.rotation_time = 2

        radar_uzwil = build_flores_monostatic_radar(
            point=Point(
                lat=47.43006,
                lon=9.11266,
                alt=self._terrain_model.elevationAt(47.43006, 9.11266),
            ),
            sensor_id=self._id_provider.increment(Entity.SENSOR, name="Uzwil radar"),
            rx_id=2,
            tx_id=2,
        )
        radar_uzwil.receiver.rotation_time = 2

        radar_immenberg = build_flores_monostatic_radar(
            point=Point(
                lat=47.53152,
                lon=8.97644,
                alt=self._terrain_model.elevationAt(47.53152, 8.97644),
            ),
            sensor_id=self._id_provider.increment(
                Entity.SENSOR, name="Immenberg radar"
            ),
            rx_id=3,
            tx_id=3,
        )
        radar_immenberg.receiver.rotation_time = 2

        # BLUE BODLUV KR.
        flak_stammerberg = self._build_bodluv(
            point=Point(
                lat=47.65007,
                lon=8.80933,
                alt=self._terrain_model.elevationAt(47.65007, 8.80933) + 10.0,
            ),
            name="Flak Stammersberg",
        )
        flak_frauenfeld = self._build_bodluv(
            point=Point(
                lat=47.55208,
                lon=8.90034,
                alt=self._terrain_model.elevationAt(47.55208, 8.90034) + 10.0,
            ),
            name="Flak Frauenfeld",
        )
        flak_cholfirst = self._build_bodluv(
            point=Point(
                lat=47.67497,
                lon=8.65944,
                alt=self._terrain_model.elevationAt(47.67497, 8.65944) + 10.0,
            ),
            name="Flak Cholfirst",
        )
        flak_lommis = self._build_bodluv(
            point=Point(
                lat=47.52726,
                lon=9.01467,
                alt=self._terrain_model.elevationAt(47.52726, 9.01467) + 10.0,
            ),
            name="Flak Lommis",
        )
        flak_diessenhofen = self._build_bodluv(
            point=Point(
                lat=47.68106,
                lon=8.75334,
                alt=self._terrain_model.elevationAt(47.68106, 8.75334) + 10.0,
            ),
            name="Flak Diessenhofen",
        )
        flak_zuzwil = self._build_bodluv(
            point=Point(
                lat=47.47685,
                lon=9.10943,
                alt=self._terrain_model.elevationAt(47.47685, 9.10943) + 10.0,
            ),
            name="Flak Zuzwil",
        )
        flak_randen = self._build_bodluv(
            point=Point(
                lat=47.76402,
                lon=8.58892,
                alt=self._terrain_model.elevationAt(47.76402, 8.58892) + 10.0,
            ),
            name="Flak Randen",
        )

        return ControllerGroup(
            [
                conf_center_controller,
                MonostaticRadarController(
                    target_id=self._id_provider.increment(
                        Entity.TARGET,
                        "Airport radar",
                    ),
                    radar=radar_airport,
                    is_blue=True,
                    rcs_model=ConstantRcsModel(rcs=3.0),
                    name="Airport radar",
                ),
                MonostaticRadarController(
                    target_id=self._id_provider.increment(
                        Entity.TARGET,
                        "Schönebärg radar",
                    ),
                    radar=radar_schoneberg,
                    is_blue=True,
                    rcs_model=ConstantRcsModel(rcs=3.0),
                    name="Schönebärg radar",
                ),
                MonostaticRadarController(
                    target_id=self._id_provider.increment(
                        Entity.TARGET,
                        "Uzwil radar",
                    ),
                    radar=radar_uzwil,
                    is_blue=True,
                    rcs_model=ConstantRcsModel(rcs=3.0),
                    name="Uzwil radar",
                ),
                MonostaticRadarController(
                    target_id=self._id_provider.increment(
                        Entity.TARGET,
                        "Immenberg radar",
                    ),
                    radar=radar_immenberg,
                    is_blue=True,
                    rcs_model=ConstantRcsModel(rcs=3.0),
                    name="Immenberg radar",
                ),
                StaticDirectFireCoordinator(
                    controllers=[
                        flak_stammerberg,
                        flak_frauenfeld,
                        flak_cholfirst,
                        flak_lommis,
                        flak_diessenhofen,
                        flak_zuzwil,
                        flak_randen,
                    ],
                    terrain=self._terrain_model,
                ),
            ]
        )

    def _get_red_controller(self) -> Controller:
        conf_center_observer = self._build_airport_observer()

        #         : 48.29187
        # Lon: 10.14691
        red_radar = build_flores_monostatic_radar(
            point=Point(
                lat=48.29187,
                lon=10.14691,
                alt=self._terrain_model.elevationAt(48.29187, 10.14691) + 10.0,
            ),
            sensor_id=self._id_provider.increment(Entity.SENSOR, "RED radar"),
            rx_id=4,
            tx_id=4,
        )

        controllers = [
            MonostaticRadarController(
                target_id=self._id_provider.increment(Entity.TARGET, name="RED Radar"),
                radar=red_radar,
                is_blue=False,
                rcs_model=ConstantRcsModel(rcs=3.0),
                name="Radar RED",
            )
        ]
        angles = self._rng.uniform(0, 90, size=(self._N_attackers,))
        offset_seconds = self._rng.uniform(0, 3600, size=(self._N_attackers,))
        for angle, t_start_seconds in zip(angles, offset_seconds, strict=True):
            controllers.append(
                self._build_single_drone(
                    angle,
                    self._t0 + datetime.timedelta(seconds=t_start_seconds),
                    name="drone",
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
