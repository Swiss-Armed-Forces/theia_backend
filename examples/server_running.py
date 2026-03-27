import datetime
import functools
from pathlib import Path
import threading

import uvicorn

from theia.data_loading import load_trajectory_file
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.logging import SituationalPictureBuffer
from theia.simulation.server import create_app
from theia.simulation.simulator import (
    Simulator,
    TimeCriterion,
    run_simulation_until_completion,
)
from theia.simulation.tracking import (
    MonostaticPseudoTracker,
)
from theia.test_data import get_uetliberg_radar

print("Initialise...")

buffer = SituationalPictureBuffer()

################################################
# Setup the simulation.
################################################

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
# stop_time = start_time + datetime.timedelta(minutes=2)
stop_time = max([t.times[-1] for t in trajectories])

time_step = datetime.timedelta(seconds=1)

simulator = Simulator(
    blue_controller=MonostaticRadarController(radar),
    red_controller=scripted_target_controller,
    # blue_tracker=MonostaticSingleSensorTracker(),
    # blue_tracker=DummyTracker(),
    blue_tracker=MonostaticPseudoTracker(),
    start_time=start_time,
    time_step=time_step,
    min_time_per_step=datetime.timedelta(seconds=1),
    termination_criterion=TimeCriterion(stop_time),
    seed=4054080,
    listener=buffer,
    simulate_clutter=False,
)

################################################
# Setup the server.
################################################
app = create_app(buffer)

################################################
# Start the simulation and the server.
################################################
sim_thread = threading.Thread(
    target=functools.partial(
        run_simulation_until_completion,
        # profile_simulation_until_completion,
        simulator=simulator,
    ),
    daemon=False,
)
sim_thread.start()

print("Start simulation...")

uvicorn.run(app, host="127.0.0.1", port=8000)
