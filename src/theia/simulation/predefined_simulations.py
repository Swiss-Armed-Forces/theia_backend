import datetime
from pathlib import Path

import numpy as np

from theia.data_loading import load_trajectory_file
from theia.detection.pcl import PclDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.pcl_sensor_controller import PclSensorController
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.logging import (
    CompositeSimulationListener,
    FileLogger,
    SituationalPictureBuffer,
)
from theia.simulation.pseudo_tracker import PseudoTracker
from theia.simulation.simulator import Simulator, TimeCriterion
from theia.simulation.tracking import MonostaticPseudoTracker
from theia.test_data import get_uetliberg_radar, load_pcl_example


def load_uetliberg_opensky_simulator() -> tuple[Simulator, SituationalPictureBuffer]:
    radar = get_uetliberg_radar(
        min_range_uncertainty=0.0,
        max_range_uncertainty=0.0,
        min_angular_uncertainty=0.0,
        max_angular_uncertainty=0.0,
    )
    radar.receiver.cpi_pulses = 1

    # Load trajectories from OpenSky.
    trajectories, _ = load_trajectory_file(
        f"{Path(__file__).resolve().parent}/../../../data/data_opensky_2022-06-27.csv"
    )

    scripted_target_controller = ControllerGroup(
        [WaypointTargetController.from_trajectory(t) for t in trajectories]
    )

    # Build the simulator.
    start_time = min([t.times[0] for t in trajectories])
    # stop_time = start_time + datetime.timedelta(minutes=2)
    stop_time = max([t.times[-1] for t in trajectories])

    time_step = datetime.timedelta(seconds=1)

    buffer = SituationalPictureBuffer()

    simulator = Simulator(
        pcl_detector=PclDetector(),
        blue_controller=MonostaticRadarController(radar),
        red_controller=scripted_target_controller,
        # blue_tracker=MonostaticSingleSensorTracker(),
        # blue_tracker=DummyTracker(),
        blue_tracker=MonostaticPseudoTracker(removal_patience=30),
        start_time=start_time,
        time_step=time_step,
        min_time_per_step=datetime.timedelta(seconds=1),
        termination_criterion=TimeCriterion(stop_time),
        rng=np.random.Generator(np.random.PCG64(seed=4054080)),
        listener=buffer,
        simulate_clutter=False,
    )

    return simulator, buffer


def load_uetliberg_single_target_simulator_pcl(
    interactive: bool = True,
) -> tuple[Simulator, SituationalPictureBuffer]:
    sensors, trajectories, grid = load_pcl_example()

    # Load trajectories from OpenSky.
    # trajectories, _ = load_trajectory_file(
    #     f"{Path(__file__).resolve().parent}/../../../data/data_opensky_2022-06-27.csv"
    # )

    scripted_target_controller = ControllerGroup(
        [WaypointTargetController.from_trajectory(t) for t in trajectories]
    )

    # Build the simulator.
    start_time = min([t.times[0] for t in trajectories])
    # stop_time = start_time + datetime.timedelta(minutes=2)
    stop_time = max([t.times[-1] for t in trajectories])

    time_step = datetime.timedelta(seconds=1)

    buffer = SituationalPictureBuffer()

    listener = buffer
    if not interactive:
        listener = CompositeSimulationListener([buffer, FileLogger("output.json")])

    rng = np.random.Generator(np.random.PCG64(seed=4054080))

    simulator = Simulator(
        pcl_detector=PclDetector(),
        blue_controller=ControllerGroup(
            [PclSensorController(sensor) for sensor in sensors]
        ),
        red_controller=scripted_target_controller,
        # blue_tracker=MonostaticSingleSensorTracker(),
        # blue_tracker=DummyTracker(),
        blue_tracker=PseudoTracker(
            removal_patience=30,
            rng=rng,
            start_time=start_time,
        ),
        start_time=start_time,
        time_step=time_step,
        min_time_per_step=datetime.timedelta(seconds=1 if interactive else 0),
        termination_criterion=TimeCriterion(stop_time),
        rng=rng,
        listener=listener,
        simulate_clutter=False,
    )

    return simulator, buffer
