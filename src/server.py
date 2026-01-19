import datetime
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from theia.data_loading import load_trajectory_file
from theia.target_simulation.constant_radar_simulator import ConstantRadarSimulator
from theia.target_simulation.recorded_targets_simulator import RecordedTargetsSimulator
from theia.types import ActiveRadarDetection, Radar, Target


data_dir = os.environ["THEIA_DIR"]

radars_file = f"{data_dir}/radars.json"
trajectories_file = f"{data_dir}/trajectories.csv"
detections_file = f"{data_dir}/detections.json"

# Load radars.
radars = []
if os.path.exists(radars_file):
    with open(radars_file, "r") as file:
        objects = json.load(file)
    radars = [Radar.model_validate(o) for o in objects]

# Load detections.
detections = []
if os.path.exists(detections_file):
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

radar_simulator = ConstantRadarSimulator(
    radars,
    target_simulator.get_minimum_time(),
    target_simulator.get_maximum_time(),
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/targets/{time}")
def get_targets(time: datetime.datetime) -> list[Target]:
    return list(target_simulator.get_targets(time))


@app.get("/radars/{time}")
def get_radars(time: datetime.datetime) -> list[Radar]:
    return list(radar_simulator.get_radars(time))


@app.get("/active_radar_detections/{time}")
def get_active_radar_detections(time: datetime.datetime) -> list[ActiveRadarDetection]:
    return [d for d in detections if d.time <= time]
