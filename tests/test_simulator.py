import datetime
from pathlib import Path
import unittest

import numpy as np
from tqdm import tqdm

from theia.data_loading import load_trajectory_file
from theia.data_loading_testing import load_pcl_reference_data
from theia.detection.pcl import PclDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.logging import FileLogger, InMemoryLogger
from theia.simulation.simulator import Simulator, TimeCriterion
from theia.simulation.tracking import DummyTracker
from theia.test_data import get_uetliberg_radar

FREQUENCY = 3_000  # Hz
POWER = 500_000  # W
DIAMETER = 4.0  # m
BANDWIDTH = 5  # MHz


class SimulatorTest(unittest.TestCase):
    def test_run_zueri_westbound(self):
        # Define monostatic radars.
        radar = get_uetliberg_radar()
        # Load targets.
        trajectories, _ = load_pcl_reference_data(
            f"{Path(__file__).resolve().parent}/test_data/pcl_detection"
        )
        scripted_target_controller = ControllerGroup(
            [WaypointTargetController.from_trajectory(t) for t in trajectories]
        )

        # Build the simulator.
        start_time = min([t.times[0] for t in trajectories])
        stop_time = max([t.times[-1] for t in trajectories])

        logger = InMemoryLogger()

        simulator = Simulator(
            pcl_detector=PclDetector(),
            blue_controller=MonostaticRadarController(radar),
            red_controller=scripted_target_controller,
            blue_tracker=DummyTracker(),
            start_time=start_time,
            time_step=datetime.timedelta(seconds=1),
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(stop_time),
            rng=np.random.Generator(np.random.PCG64(seed=4054080)),
            listener=logger,
        )
        # Simulate until the end.
        while simulator.advance():
            pass

    def test_opensky(self):
        radar = get_uetliberg_radar()

        # Load trajectories from OpenSky.
        trajectories, _ = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/../data/data_opensky_2022-06-27.csv"
        )

        scripted_target_controller = ControllerGroup(
            [WaypointTargetController.from_trajectory(t) for t in trajectories]
        )

        # Build the simulator.
        start_time = min([t.times[0] for t in trajectories])
        stop_time = max([t.times[-1] for t in trajectories])

        logger = FileLogger("log_opensky.json")

        time_step = datetime.timedelta(seconds=1)

        simulator = Simulator(
            pcl_detector=PclDetector(),
            blue_controller=MonostaticRadarController(radar),
            red_controller=scripted_target_controller,
            blue_tracker=DummyTracker(),
            start_time=start_time,
            time_step=time_step,
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(stop_time),
            rng=np.random.Generator(np.random.PCG64(seed=4054080)),
            listener=logger,
        )
        n_iterations = int(
            np.ceil((stop_time - start_time).seconds / time_step.seconds)
        )

        # Simulate until the end.
        for _ in tqdm(range(n_iterations)):
            is_success = simulator.advance()
            self.assertTrue(is_success)
        self.assertFalse(simulator.advance())


if __name__ == "__main__":
    unittest.main()
