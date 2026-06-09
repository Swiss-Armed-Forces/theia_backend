import datetime
from pathlib import Path

from theia.config import SIDC_RED_FIXED_WING
from theia.data_loading import load_trajectory_file
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.factories.abstract_simulator_factory import (
    AbstractSimulatorFactory,
)
from theia.simulation.simulator import TerminationCriterion, TimeCriterion
from theia.simulation.trackers.tracking import MonostaticPseudoTracker
from theia.test_data import get_uetliberg_radar
from theia.types import AbstractTracker, ConstantRcsModel, Controller


class UetlibergOpenskySimulatorFactory(AbstractSimulatorFactory):
    def __init__(self):
        self._radar = get_uetliberg_radar(
            min_range_uncertainty=0.0,
            max_range_uncertainty=0.0,
            min_angular_uncertainty=0.0,
            max_angular_uncertainty=0.0,
        )
        self._radar.receiver.cpi_pulses = 1
        self._trajectories, _ = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/../../../../data/data_opensky_2022-06-27.csv"
        )

    def _get_pcl_detector(self) -> PclDetector:
        return PclDetector()

    def _get_pet_detector(self) -> PetDetector:
        return PetDetector()

    def _get_blue_tracker(self) -> AbstractTracker:
        return MonostaticPseudoTracker(removal_patience=30)

    def _get_red_tracker(self) -> AbstractTracker:
        return MonostaticPseudoTracker(removal_patience=30)

    def _get_blue_controller(self) -> Controller:
        return MonostaticRadarController(
            target_id=max([t.target_id for t in self._trajectories]) + 1,
            radar=self._radar,
            is_blue=True,
            rcs_model=ConstantRcsModel(rcs=1.0),
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
