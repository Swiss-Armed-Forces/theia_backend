import abc
import datetime

import numpy as np

from theia.config import SIDC_RED_FIXED_WING
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.pcl_sensor_controller import PclSensorController
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.factories.abstract_simulator_factory import (
    AbstractSimulatorFactory,
)
from theia.simulation.pseudo_tracker import PseudoTracker
from theia.simulation.simulator import TerminationCriterion, TimeCriterion
from theia.test_data import load_pcl_example
from theia.types import AbstractTracker, Controller


class UetlibergPclSimulatorFactory(AbstractSimulatorFactory):
    def __init__(
        self,
        rng: np.random.Generator,
        bistatic_range_uncertainty: float = 0.0,
    ):
        self._rng = rng
        self._bistatic_range_uncertainty = bistatic_range_uncertainty

        self._sensors, self._trajectories, grid = load_pcl_example(
            bistatic_range_uncertainty=self._bistatic_range_uncertainty,
        )

    def _get_pcl_detector(self) -> PclDetector:
        return PclDetector()

    def _get_pet_detector(self) -> PetDetector:
        return PetDetector()

    def _get_blue_tracker(self) -> AbstractTracker:
        return PseudoTracker(
            removal_patience=30,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_red_tracker(self) -> AbstractTracker:
        PseudoTracker(
            removal_patience=30,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_blue_controller(self) -> Controller:
        return ControllerGroup(
            [PclSensorController(sensor) for sensor in self._sensors]
        )

    def _get_red_controller(self) -> Controller:
        return ControllerGroup(
            [
                WaypointTargetController.from_trajectory(t, SIDC_RED_FIXED_WING)
                for t in self._trajectories
            ]
        )

    def _get_start_time(self) -> datetime.datetime:
        return min([t.times[0] for t in self._trajectories])

    def _get_timestep(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=1)

    def _get_termination_criterion(self) -> TerminationCriterion:
        stop_time = max([t.times[-1] for t in self._trajectories])
        return TimeCriterion(stop_time)
