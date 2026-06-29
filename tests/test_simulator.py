import datetime
from pathlib import Path
import unittest

import numpy as np
from tqdm import tqdm

from theia.config import SIDC
from theia.data_loading import load_trajectory_file
from theia.data_loading_testing import load_pcl_reference_data
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.damage_model import UniformDamageModel
from theia.simulation.logging import FileLogger, InMemoryLogger
from theia.simulation.simulator import Simulator, TimeCriterion
from theia.simulation.trackers.tracking import DummyTracker
from theia.terrain import SrtmTerrainModel
from theia.test_data import get_uetliberg_radar
from theia.types import ConstantRcsModel

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
            [
                WaypointTargetController.from_trajectory(t, SIDC.RED_FIXED_WING)
                for t in trajectories
            ]
        )

        # Build the simulator.
        start_time = min([t.times[0] for t in trajectories])
        stop_time = max([t.times[-1] for t in trajectories])

        logger = InMemoryLogger()

        terrain = SrtmTerrainModel()

        rng = np.random.Generator(np.random.PCG64(seed=4054080))

        simulator = Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=MonostaticRadarController(
                len(scripted_target_controller._controllers) + 1,
                radar,
                True,
                ConstantRcsModel(rcs=1.0),
            ),
            red_controller=scripted_target_controller,
            blue_tracker=DummyTracker(),
            red_tracker=DummyTracker(),
            start_time=start_time,
            time_step=datetime.timedelta(seconds=1),
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(stop_time),
            rng=rng,
            listener=logger,
            terrain_model=terrain,
            damage_model=UniformDamageModel(1.0, rng),
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
            [
                WaypointTargetController.from_trajectory(t, SIDC.RED_FIXED_WING)
                for t in trajectories
            ]
        )

        # Build the simulator.
        start_time = min([t.times[0] for t in trajectories])
        stop_time = max([t.times[-1] for t in trajectories])

        logger = FileLogger("log_opensky.json")

        time_step = datetime.timedelta(seconds=1)

        terrain = SrtmTerrainModel()

        rng = np.random.Generator(np.random.PCG64(seed=4054080))

        simulator = Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=MonostaticRadarController(
                len(scripted_target_controller._controllers) + 1,
                radar,
                True,
                ConstantRcsModel(rcs=1.0),
            ),
            red_controller=scripted_target_controller,
            blue_tracker=DummyTracker(),
            red_tracker=DummyTracker(),
            start_time=start_time,
            time_step=time_step,
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(stop_time),
            rng=rng,
            listener=logger,
            terrain_model=terrain,
            damage_model=UniformDamageModel(1.0, rng),
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
