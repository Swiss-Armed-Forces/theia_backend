import os
from pathlib import Path

import numpy as np
from theia.data_loading import load_trajectory_file
from theia.scenario_simulator import ScenarioSimulator
from theia.target_simulation.constant_radar_simulator import ConstantRadarSimulator
from theia.target_simulation.recorded_targets_simulator import RecordedTargetsSimulator
from theia.types import Point, Polarization, Radar

# Setup.
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

trajectories, callsign_lookup = load_trajectory_file(
    f"{Path(__file__).resolve().parent}/../data/data_opensky_2022-06-27.csv"
)

for trajectory in trajectories:
    trajectory.cross_sections = np.ones(len(trajectory.times)) * 10.0

trajectories = [trajectory for trajectory in trajectories if len(trajectory.times) > 1]

target_simulator = RecordedTargetsSimulator(trajectories)
radar_simulator = ConstantRadarSimulator(
    radars=[radar],
    on_time=target_simulator.get_minimum_time(),
    off_time=target_simulator.get_maximum_time(),
)

rng = np.random.Generator(np.random.PCG64(seed=45797093))
simulator = ScenarioSimulator(target_simulator, radar_simulator, rng)

detections = simulator.simulate_active_radar_detections()
