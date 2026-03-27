import datetime
import functools
from pathlib import Path
import threading

import uvicorn

from theia.coordinates import POSITIONS_OF_INTEREST
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
    profile_simulation_until_completion,
    run_simulation_until_completion,
)
from theia.simulation.tracking import (
    DummyTracker,
    MonostaticPseudoTracker,
    MonostaticSingleSensorTracker,
)
from theia.types import (
    MonostaticRadarMeasurementModel,
    Point,
    Polarization,
    Radar,
    Receiver,
    Transmitter,
)

print("Initialise...")

buffer = SituationalPictureBuffer()

################################################
# Setup the simulation.
################################################

FREQUENCY = 3_000  # Hz
POWER = 500_000  # W
DIAMETER = 4.0  # m
BANDWIDTH = 5  # MHz

radar_lat = POSITIONS_OF_INTEREST["Uetliberg"]["lat"]
radar_lon = POSITIONS_OF_INTEREST["Uetliberg"]["lon"]
radar_alt = POSITIONS_OF_INTEREST["Uetliberg"]["alt"]

radar = Radar(
    transmitter=Transmitter(
        id=0,
        point=Point(lat=radar_lat, lon=radar_lon, alt=radar_alt),
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
        point=Point(lat=radar_lat, lon=radar_lon, alt=radar_alt),
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
    error_model=MonostaticRadarMeasurementModel(
        min_range_uncertainty=0.0,
        max_range_uncertainty=0.0,
        min_angular_uncertainty=0.0,
        max_angular_uncertainty=0.0,
    ),
)

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
