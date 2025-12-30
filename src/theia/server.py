import datetime
import json
import os
from fastapi import FastAPI
import numpy as np

from theia.data_loading import load_trajectory_file
from theia.target_simulation.constant_radar_simulator import ConstantRadarSimulator
from theia.target_simulation.recorded_targets_simulator import RecordedTargetsSimulator
from theia.types import ActiveRadarDetection, Point, Polarization, Radar, Target


trajectories_file = os.environ["THEIA_TRAJECTORY_FILE"]
detections_file = os.environ.get("THEIA_DETECTIONS_FILE", None)

# Load detections.
detections = []
if detections_file is not None:
    with open(detections_file, "r") as file:
        objects = json.load(file)
    detections = [ActiveRadarDetection.model_validate(o) for o in objects]

# Load target trajectories.
# Sanitise missing data.
# TODO: Remove after development.
DEFAULT_CROSS_SECTION = 10.0
trajectories, callsign_lookup = load_trajectory_file(trajectories_file)
for trajectory in trajectories:
    if np.isnan(trajectory.cross_sections).all():
        trajectory.cross_sections = (
            np.ones(len(trajectory.times)) * DEFAULT_CROSS_SECTION
        )
trajectories = [trajectory for trajectory in trajectories if len(trajectory.times) > 1]
target_simulator = RecordedTargetsSimulator(trajectories)

# Load radar positions.
# TODO: Do not hardcode!
radar = Radar(
    id=585,
    point=Point(
        lat=47.36700085728634,
        lon=8.537724304199216,
        alt=408,
    ),
    power=20000,
    erp=800,
    antenna_height=10.0,
    diameter=2.0,
    frequency=1000.0,
    pulse_width=1,
    cpi_pulses=1,
    bandwidth=1,
    pfa=1e-6,
    min_elevation=-20.0,
    max_elevation=60.0,
    rotation_time=10.0,
    polarization=Polarization.HORIZONTAL,
)
radar_simulator = ConstantRadarSimulator(
    [radar],
    target_simulator.get_minimum_time(),
    target_simulator.get_maximum_time(),
)

app = FastAPI()


@app.get("/targets/{time}")
def get_targets(time: datetime.datetime) -> list[Target]:
    return list(target_simulator.get_targets(time))


@app.get("/radars/{time}")
def get_radars(time: datetime.datetime) -> list[Radar]:
    return list(radar_simulator.get_radars(time))


@app.get("/active_radar_detections/{time}")
def get_active_radar_detections(time: datetime.datetime) -> list[ActiveRadarDetection]:
    return [d for d in detections if d.time <= time]
