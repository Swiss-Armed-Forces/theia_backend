import datetime
import json
from pathlib import Path
import unittest

import numpy as np
from tqdm import tqdm

from theia.data_loading import load_trajectory_file
from theia.data_loading_testing import load_pcl_reference_data
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
from theia.types import MonostaticRadarMeasurementModel, Point, Polarization, Radar, Receiver, Transmitter

FREQUENCY = 3_000 # Hz
POWER = 500_000 # W
DIAMETER = 4. # m
BANDWIDTH = 5 # MHz


class SimulatorTest(unittest.TestCase):
    def test_run_zueri_westbound(self):
        # Define monostatic radars.
        radar = Radar(
            transmitter=Transmitter(
                id=0,
                point=Point(lat=47.349491, lon=8.492063, alt=856.2037851199802),
                power=POWER,
                erp=1000.0,
                antenna_height=10.0,
                antenna_diameter=DIAMETER,
                frequency=FREQUENCY,
                pulse_width=1.0,
                polarization=Polarization.VERTICAL,
                bandwidth=BANDWIDTH,
                max_coherent_integration_time=0.5,
                antenna_efficiency_value=0.6,
                vertical_attenuation=None,
                horizontal_attenuation=None,
            ),
            receiver=Receiver(
                id=0,
                point=Point(lat=47.349491, lon=8.492063, alt=856.2037851199802),
                antenna_height=10.0,
                diameter=DIAMETER,
                cpi_pulses=1.0,
                pfa=1e-06,
                min_elevation=-20.0,
                max_elevation=60.0,
                rotation_time=10.0,
                bandwidth=BANDWIDTH,
                gain=0,
                losses=0,
                noise_temperature=300.0,
                noise_figure=1.9,
                antenna_efficiency_value=0.6,
                vertical_attenuation=None,
                horizontal_attenuation=None,
            ),
            error_model=MonostaticRadarMeasurementModel(),
        )
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
            blue_controller=MonostaticRadarController(radar),
            red_controller=scripted_target_controller,
            blue_tracker=DummyTracker(),
            start_time=start_time,
            time_step=datetime.timedelta(seconds=1),
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(stop_time),
            seed=4054080,
            listener=logger,
        )
        # Simulate until the end.
        while simulator.advance():
            pass
    
    def test_opensky(self):
        radar = Radar(
            transmitter=Transmitter(
                id=0,
                point=Point(lat=47.349491, lon=8.492063, alt=856.2037851199802),
                power=POWER,
                erp=1000.0,
                antenna_height=10.0,
                antenna_diameter=DIAMETER,
                frequency=FREQUENCY,
                pulse_width=1.0,
                polarization=Polarization.VERTICAL,
                bandwidth=BANDWIDTH,
                max_coherent_integration_time=0.5,
                antenna_efficiency_value=0.6,
                vertical_attenuation=None,
                horizontal_attenuation=None,
            ),
            receiver=Receiver(
                id=0,
                point=Point(lat=47.349491, lon=8.492063, alt=856.2037851199802),
                antenna_height=10.0,
                diameter=DIAMETER,
                cpi_pulses=1.0,
                pfa=1e-06,
                min_elevation=-20.0,
                max_elevation=60.0,
                rotation_time=10.0,
                bandwidth=BANDWIDTH,
                gain=0,
                losses=0,
                noise_temperature=300.0,
                noise_figure=1.9,
                antenna_efficiency_value=0.6,
                vertical_attenuation=None,
                horizontal_attenuation=None,
            ),
            error_model=MonostaticRadarMeasurementModel(),
        )

        # Load trajectories from OpenSky.
        trajectories, _ = load_trajectory_file(f"{Path(__file__).resolve().parent}/../data/data_opensky_2022-06-27.csv")

        scripted_target_controller = ControllerGroup(
            [WaypointTargetController.from_trajectory(t) for t in trajectories]
        )

        # Build the simulator.
        start_time = min([t.times[0] for t in trajectories])
        stop_time = max([t.times[-1] for t in trajectories])

        logger = FileLogger("log_opensky.json")

        time_step = datetime.timedelta(seconds=1)

        simulator = Simulator(
            blue_controller=MonostaticRadarController(radar),
            red_controller=scripted_target_controller,
            blue_tracker=DummyTracker(),
            start_time=start_time,
            time_step=time_step,
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(stop_time),
            seed=4054080,
            listener=logger,
        )
        n_iterations = int(np.ceil((stop_time - start_time).seconds / time_step.seconds))

        # Simulate until the end.
        for _ in tqdm(range(n_iterations)):
            is_success = simulator.advance()
            self.assertTrue(is_success)
        self.assertFalse(simulator.advance())


if __name__ == "__main__":
    unittest.main()
