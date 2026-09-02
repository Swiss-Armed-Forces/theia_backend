import datetime

import numpy as np

from theia.config import SIDC
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.effectors import DirectFireEffector
from theia.maneuvers import ConstantSpeedStraightManeuver
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.static_direct_fire_controller import (
    StaticDirectFireController,
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
from theia.test_data import get_uetliberg_radar
from theia.types import AbstractTracker, ConstantRcsModel, Controller, Point


class SingleTargetSingleEffectorFactory(AbstractSimulatorFactory):
    def __init__(
        self,
        rng: np.random.Generator,
        terrain_model: AbstractTerrainModel,
        include_uncertainty: bool,
    ):
        self._rng = rng
        self._terrain = terrain_model
        self._t0 = datetime.datetime.fromtimestamp(0)

        ANGULAR_UNCERTAINTY = np.deg2rad(0.5) if include_uncertainty else 0.0
        RANGE_UNCERTAINTY = 50.0 if include_uncertainty else 0.0

        # Build BLUE sensor.
        self._radar = get_uetliberg_radar(
            min_range_uncertainty=RANGE_UNCERTAINTY,
            max_range_uncertainty=RANGE_UNCERTAINTY,
            min_angular_uncertainty=ANGULAR_UNCERTAINTY,
            max_angular_uncertainty=ANGULAR_UNCERTAINTY,
        )
        self._radar.receiver.cpi_pulses = 1

        # Build RED plane.
        p_start = Point(lat=47.4002, lon=8.6381, alt=800)
        p_stop = Point(lat=47.3088, lon=8.4240, alt=800.0)

        self._times, self._points = ConstantSpeedStraightManeuver(
            stop_point=p_stop, speed=300.0
        ).get_waypoints(
            self._t0,
            p_start,
        )

        self._target_id_radar = 0
        self._target_id_plane = 1

    def _get_pcl_detector(self) -> PclDetector:
        return PclDetector()

    def _get_pet_detector(self) -> PetDetector:
        return PetDetector(terrain_model=self._terrain)

    def _get_blue_tracker(self) -> AbstractTracker:
        return PseudoTracker(
            removal_patience=30,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_red_tracker(self) -> AbstractTracker:
        return PseudoTracker(
            removal_patience=30,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_blue_controller(self) -> Controller:
        radar_controller = MonostaticRadarController(
            target_id=self._target_id_radar,
            radar=self._radar,
            is_blue=True,
            rcs_model=ConstantRcsModel(rcs=1.0),
        )
        effector_controller = StaticDirectFireController(
            target_id=20,
            sidc=SIDC.BLUE_AIR_DEFENCE,
            rcs=1.0,
            effector=DirectFireEffector(
                id=0,
                name="my effector",
                point=self._radar.receiver.point,
                combat_range=5_000,
                n_attacks_left=1,
                terrain=self._terrain,
                cadence=1.0,
            ),
            assigned_track_id="0",
        )

        return ControllerGroup([radar_controller, effector_controller])

    def _get_red_controller(self) -> Controller:
        return LivingController(
            child=WaypointTargetController(
                name="RED plane",
                sidc=SIDC.RED_FIXED_WING,
                target_id=self._target_id_plane,
                times=self._times,
                waypoints=self._points,
                rcs_model=ConstantRcsModel(rcs=1.0),
            ),
            target_id=self._target_id_plane,
        )

    def _get_start_time(self) -> datetime.datetime:
        return self._t0

    def _get_timestep(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=1)

    def _get_termination_criterion(self) -> TerminationCriterion:
        stop_time = self._times[-1]
        return TimeCriterion(stop_time)
