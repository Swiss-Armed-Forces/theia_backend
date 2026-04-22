import cProfile
import datetime

import numpy as np
from tqdm import tqdm

from theia.detection.pcl import PclDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.pcl_sensor_controller import PclSensorController
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.logging import NoLogger
from theia.simulation.pseudo_tracker import PseudoTracker
from theia.simulation.simulator import Simulator, TimeCriterion
from theia.test_data import load_pcl_example


sensors, trajectories, grid = load_pcl_example()
trajectory = trajectories[0]
RCS = trajectory.cross_section_model.rcs

scripted_target_controller = ControllerGroup(
    [WaypointTargetController.from_trajectory(t) for t in trajectories]
)
pcl_detector = PclDetector()

# Build the simulator.
start_time = min([t.times[0] for t in trajectories])
stop_time = max([t.times[-1] for t in trajectories])

time_step = datetime.timedelta(seconds=1)

rng = np.random.Generator(np.random.PCG64(seed=4054080))

simulator = Simulator(
    pcl_detector=pcl_detector,
    blue_controller=ControllerGroup(
        [PclSensorController(sensor) for sensor in sensors]
    ),
    red_controller=scripted_target_controller,
    blue_tracker=PseudoTracker(removal_patience=30, rng=rng),
    start_time=start_time,
    time_step=time_step,
    min_time_per_step=datetime.timedelta(seconds=0),
    termination_criterion=TimeCriterion(stop_time),
    rng=rng,
    listener=NoLogger(),
    simulate_clutter=False,
)

profiler = cProfile.Profile()
profiler.enable()

for _ in tqdm(range(100)):
    simulator.advance()

profiler.disable()
profiler.dump_stats("profile__simulator_advance.prof")
