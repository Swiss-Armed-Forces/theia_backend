import datetime
from pathlib import Path

import numpy as np

from theia.config import SIDC_RED_FIXED_WING
from theia.coordinates import POSITIONS_OF_INTEREST
from theia.data_loading import load_bakom_ukw_transmitters, load_trajectory_file
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
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
from theia.terrain import elevationAt
from theia.test_data import (
    build_fighter_jet_radar,
    build_flores_monostatic_radar,
    build_pcl_receiver,
    build_single_target_from_Bodensee,
    get_uetliberg_radar,
    load_pcl_example,
)
from theia.types import (
    ConstantRcsModel,
    MonostaticSensor,
    PclMeasurementModel,
    PclSensor,
    Point,
)


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
        [
            WaypointTargetController.from_trajectory(t, SIDC_RED_FIXED_WING)
            for t in trajectories
        ]
    )

    # Build the simulator.
    start_time = min([t.times[0] for t in trajectories])
    # stop_time = start_time + datetime.timedelta(minutes=2)
    stop_time = max([t.times[-1] for t in trajectories])

    time_step = datetime.timedelta(seconds=1)

    buffer = SituationalPictureBuffer()

    simulator = Simulator(
        pcl_detector=PclDetector(),
        pet_detector=PetDetector(),
        blue_controller=MonostaticRadarController(
            target_id=max([t.target_id for t in trajectories]) + 1,
            radar=radar,
            is_blue=True,
            rcs_model=ConstantRcsModel(rcs=1.0),
        ),
        red_controller=scripted_target_controller,
        # blue_tracker=MonostaticSingleSensorTracker(),
        # blue_tracker=DummyTracker(),
        blue_tracker=MonostaticPseudoTracker(removal_patience=30),
        red_tracker=MonostaticPseudoTracker(removal_patience=30),
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
    bistatic_range_uncertainty: float = 0.0,
) -> tuple[Simulator, SituationalPictureBuffer]:
    sensors, trajectories, grid = load_pcl_example(
        bistatic_range_uncertainty=bistatic_range_uncertainty,
    )

    # Load trajectories from OpenSky.
    # trajectories, _ = load_trajectory_file(
    #     f"{Path(__file__).resolve().parent}/../../../data/data_opensky_2022-06-27.csv"
    # )

    scripted_target_controller = ControllerGroup(
        [
            WaypointTargetController.from_trajectory(t, SIDC_RED_FIXED_WING)
            for t in trajectories
        ]
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
        red_tracker=PseudoTracker(
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


def load_single_target_from_Bodensee_simulator(
    interactive: bool = True,
) -> tuple[Simulator, SituationalPictureBuffer]:
    # Build sensors.
    monostatic_positions = [
        (
            POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
            POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
        ),
        # (46.97947680813955, 8.254819833983184),  # Pilatus
        # (46.572082512448134, 8.829917042912648),  # Scopi
        # (46.101596555481564, 7.716021211924495),  # Weisshorn
        # (46.835925325283476, 9.794288419787135),  # Weissfluh
    ]
    radar_lat = POSITIONS_OF_INTEREST["Uetliberg"]["lat"]
    radar_lon = POSITIONS_OF_INTEREST["Uetliberg"]["lon"]
    point = Point(lat=radar_lat, lon=radar_lon, alt=elevationAt(radar_lat, radar_lon))
    sensor_id = 0
    monostatic_radars: list[MonostaticSensor] = []
    for pos in monostatic_positions:
        point = Point(lat=pos[0], lon=pos[1], alt=elevationAt(pos[0], pos[1]))
        monostatic_radar = build_flores_monostatic_radar(point, sensor_id, 0, 0)
        monostatic_radars.append(monostatic_radar)
        sensor_id += 1

    pcl_rx_lat = 46.99166835
    pcl_rx_lon = 8.36833333
    point = Point(
        lat=pcl_rx_lat,
        lon=pcl_rx_lon,
        alt=elevationAt(pcl_rx_lat, pcl_rx_lon),
    )
    pcl_rx = build_pcl_receiver(rx_id=1, point=point)
    txs = load_bakom_ukw_transmitters(start_id=1)
    tx_ids = [943, 1151, 921, 1113]
    txs = [tx for tx in txs if tx.id in tx_ids]
    pcl_sensors: list[PclSensor] = []
    for tx in txs:
        pcl_sensors.append(
            PclSensor(
                id=sensor_id,
                transmitter=tx,
                receiver=pcl_rx,
                error_model=PclMeasurementModel(
                    min_bistatic_range_uncertainty=200.0,
                    max_bistatic_range_uncertainty=200.0,
                    min_doppler_uncertainty=5.0,
                    max_doppler_uncertainty=5.0,
                ),
            )
        )
        sensor_id += 1

    # Load trajectory.
    trajectory = build_single_target_from_Bodensee(alt=1000)
    trajectories = [trajectory]

    scripted_target_controller = ControllerGroup(
        [
            WaypointTargetController.from_trajectory(
                t,
                SIDC_RED_FIXED_WING,
                sensor=build_fighter_jet_radar(1, 1, 1),
            )
            for t in trajectories
        ]
    )

    # Build controllers for the sensors.
    monostatic_controllers = [
        MonostaticRadarController(1 + i, r, True, ConstantRcsModel(rcs=1.0))
        for i, r in enumerate(monostatic_radars)
    ]
    pcl_controllers = [
        PclSensorController(
            monostatic_controllers[-1]._target_id + 2 * i + 1,
            monostatic_controllers[-1]._target_id + 2 * i + 2,
            sensor,
            True,
            ConstantRcsModel(rcs=1.0),
        )
        for i, sensor in enumerate(pcl_sensors)
    ]
    blue_controller = ControllerGroup(monostatic_controllers + pcl_controllers)

    # Build the simulator.
    start_time = min([t.times[0] for t in trajectories])
    stop_time = max([t.times[-1] for t in trajectories])

    time_step = datetime.timedelta(seconds=1)

    buffer = SituationalPictureBuffer()

    rng = np.random.Generator(np.random.PCG64(seed=4054080))

    simulator = Simulator(
        pcl_detector=PclDetector(),
        pet_detector=PetDetector(),
        blue_controller=blue_controller,
        red_controller=scripted_target_controller,
        blue_tracker=PseudoTracker(
            removal_patience=30,
            rng=rng,
            start_time=start_time,
        ),
        red_tracker=PseudoTracker(
            removal_patience=30,
            rng=rng,
            start_time=start_time,
        ),
        start_time=start_time,
        time_step=time_step,
        min_time_per_step=datetime.timedelta(seconds=1 if interactive else 0),
        termination_criterion=TimeCriterion(stop_time),
        rng=rng,
        listener=buffer,
        simulate_clutter=False,
    )

    return simulator, buffer


def build_single_radar_single_target(
    interactive: bool = True,
) -> tuple[Simulator, SituationalPictureBuffer]:
    """
    Build a simulation for a simple scenario:
    
    There is one RED fighter jet approaching in a curved trajectory from Bodensee
    at 300m/s, equipped with a radar. Further, there is one BLUE monostatic
    radar on top of Üetliberg.
    """
    red_sensor = build_fighter_jet_radar(0, 0, 0)
    red_trajectory = build_single_target_from_Bodensee(target_id=0)
    red_controller = WaypointTargetController.from_trajectory(
        trajectory=red_trajectory,
        sidc="10060100001101000000",
        name="Reconnaissance plane",
        sensor=red_sensor,
    )

    blue_sensor = build_flores_monostatic_radar(
        Point(
            lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
            lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
            alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
        ),
        1,
        1,
        1,
    )
    blue_controller = MonostaticRadarController(
        target_id=1,
        radar=blue_sensor,
        is_blue=True,
        rcs_model=ConstantRcsModel(rcs=1.0),
        name="Üetliberg radar",
    )

    # Build the simulator.
    start_time = red_trajectory.times[0]
    stop_time = red_trajectory.times[-1]

    time_step = datetime.timedelta(seconds=1)

    buffer = SituationalPictureBuffer()

    rng = np.random.Generator(np.random.PCG64(seed=4054080))

    simulator = Simulator(
        pcl_detector=PclDetector(),
        pet_detector=PetDetector(),
        blue_controller=blue_controller,
        red_controller=red_controller,
        blue_tracker=PseudoTracker(
            removal_patience=30,
            rng=rng,
            start_time=start_time,
        ),
        red_tracker=PseudoTracker(
            removal_patience=30,
            rng=rng,
            start_time=start_time,
        ),
        start_time=start_time,
        time_step=time_step,
        min_time_per_step=datetime.timedelta(seconds=1 if interactive else 0),
        termination_criterion=TimeCriterion(stop_time),
        rng=rng,
        listener=buffer,
        simulate_clutter=False,
    )

    return simulator, buffer