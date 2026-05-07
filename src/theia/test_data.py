import copy
import datetime
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline
from theia.config import SIDC_RED_FIXED_WING, SIDC_UNKNOWN
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.data_loading import load_bakom_ukw_transmitters
from theia.distance import line_of_sight_distance
from theia.grids import LatLonHeightGrid
from theia.maneuvers import ConstantSpeedCurveManeuver
from theia.terrain import elevationAt
from theia.types import (
    ConstantRcsModel,
    MonostaticRadarMeasurementModel,
    MonostaticSensor,
    PclMeasurementModel,
    PclSensor,
    Point,
    Polarization,
    Receiver,
    Target,
    Trajectory,
    Transmitter,
    calculate_antenna_gain,
)
from theia.util import frequency_to_wavelength, from_dB


class TestSituationLoader:
    @staticmethod
    def load_zuerich_single_target(
        speed: float,
        move_direction: Literal["north"]
        | Literal["east"]
        | Literal["south"]
        | Literal["west"] = "south",
    ) -> tuple[list[MonostaticSensor], list[Target]]:
        radar_lat = POSITIONS_OF_INTEREST["Uetliberg"]["lat"]
        radar_lon = POSITIONS_OF_INTEREST["Uetliberg"]["lon"]
        point = Point(
            lat=radar_lat, lon=radar_lon, alt=elevationAt(radar_lat, radar_lon)
        )

        radar = get_uetliberg_radar(
            frequency=1000.0,
            diameter=2.0,
            rx_bandwidth=100,
        )

        p_target = copy.deepcopy(point)
        p_target.lat += 0.1

        if move_direction == "north":
            v = CoordinateTransformations.velocity_geodetic_to_cartesian(
                p_target, -speed, 0.0, 0.0
            )
        elif move_direction == "east":
            v = CoordinateTransformations.velocity_geodetic_to_cartesian(
                p_target, 0.0, speed, 0.0
            )
        elif move_direction == "south":
            v = CoordinateTransformations.velocity_geodetic_to_cartesian(
                p_target, speed, 0.0, 0.0
            )
        elif move_direction == "west":
            v = CoordinateTransformations.velocity_geodetic_to_cartesian(
                p_target, 0.0, -speed, 0.0
            )
        else:
            raise RuntimeError(f"Unknown direction: {move_direction}")

        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC_UNKNOWN,
            point=p_target,
            cross_section_model=ConstantRcsModel(rcs=2.0),
            velocity=v,
        )

        return [radar], [target]


def get_uetliberg_radar(
    frequency: float = 3_000,
    power: float = 50_000,
    diameter: float = 4.0,
    rx_bandwidth: float = 5.0,
    tx_bandwidth: float | None = None,
    rotation_time: float = 10.0,
    integration_time: float = 0.1,
    min_range_uncertainty: float = 100,
    max_range_uncertainty: float = float(10_000),
    min_angular_uncertainty: float = float(np.deg2rad(1)),
    max_angular_uncertainty: float = float(np.deg2rad(360)),
) -> MonostaticSensor:
    """
    Parameters
    ----------
    frequency: float, default 3000.0
        Frequency [Hz]
    power: float, default 500000.0
        Power [W]
    diameter: float, default 4.0
        Diameter [m]
    bandwidth: float, default 5.0
        Bandwidth [MHz]
    tx_bandwidth: float | None, default None
        Bandwidth of the transmitter [MHz]
        If None (default), the receiver's bandwidth is assumed
    rotation_time: float, default 10.0
        Time between detections for each receiver [s]
    integration_time: float, default 1 / (5 * 1e6)
        Integration time of the receiver [s]
    min_range_uncertainty: float, default 100
        Minimum range uncertainty for the error model [m]
    max_range_uncertainty: float, default np.inf
        Maximum range uncertainty for the error model [m]
    min_angular_uncertainty: float, default np.deg2rad(1)
        Minimum angular uncertainty for the error model [rad]
    max_angular_uncertainty: float, default np.deg2rad(360)
        Maximum angular uncertainty for the error model [rad]
    """

    radar_lat = POSITIONS_OF_INTEREST["Uetliberg"]["lat"]
    radar_lon = POSITIONS_OF_INTEREST["Uetliberg"]["lon"]
    radar_alt = POSITIONS_OF_INTEREST["Uetliberg"]["alt"]
    antenna_efficiency = 0.6

    if tx_bandwidth is None:
        tx_bandwidth = rx_bandwidth

    return MonostaticSensor(
        id=0,
        transmitter=Transmitter(
            id=0,
            point=Point(lat=radar_lat, lon=radar_lon, alt=radar_alt),
            power=power,
            erp=1000.0,
            antenna_height=10.0,
            antenna_diameter=diameter,
            antenna_gain=calculate_antenna_gain(
                diameter,
                frequency_to_wavelength(frequency),
                efficiency_value=antenna_efficiency,
            ),
            frequency=frequency,
            pulse_width=1.0,
            polarization=Polarization.VERTICAL,
            bandwidth=rx_bandwidth,
            max_coherent_integration_time=0.5,
            antenna_efficiency_value=antenna_efficiency,
            vertical_attenuation=None,
            horizontal_attenuation=None,
        ),
        receiver=Receiver(
            id=0,
            point=Point(lat=radar_lat, lon=radar_lon, alt=radar_alt),
            antenna_height=10.0,
            diameter=diameter,
            cpi_pulses=int(np.floor(integration_time * tx_bandwidth * 1e6)),
            pfa=1e-06,
            min_elevation=-20.0,
            max_elevation=60.0,
            rotation_time=rotation_time,
            bandwidth=rx_bandwidth,
            gain=0,
            losses=0,
            noise_temperature=300.0,
            noise_figure=1.9,
            antenna_efficiency_value=0.6,
            vertical_attenuation=None,
            horizontal_attenuation=None,
        ),
        error_model=MonostaticRadarMeasurementModel(
            min_range_uncertainty=min_range_uncertainty,
            max_range_uncertainty=max_range_uncertainty,
            min_angular_uncertainty=min_angular_uncertainty,
            max_angular_uncertainty=max_angular_uncertainty,
        ),
    )


def load_pcl_example(
    rcs: float = 1.0,
    rotation_time: float = 1.0,
    bistatic_range_uncertainty: float = 0.0,
) -> tuple[list[PclSensor], list[Trajectory], LatLonHeightGrid]:
    # Define the sensors.
    rx = get_uetliberg_radar(
        min_range_uncertainty=0.0,
        max_range_uncertainty=0.0,
        min_angular_uncertainty=0.0,
        max_angular_uncertainty=0.0,
        rotation_time=rotation_time,
    ).receiver
    rx.noise_figure = 0

    transmitters = load_bakom_ukw_transmitters()
    transmitters = list(
        filter(
            lambda tx: (
                line_of_sight_distance(*rx.point.as_tuple(), *tx.point.as_tuple())
                <= 20_000
            ),
            transmitters,
        )
    )

    pcl_sensors: list[PclSensor] = []
    for i, tx in enumerate(transmitters):
        rx = get_uetliberg_radar(
            rx_bandwidth=tx.bandwidth,
            tx_bandwidth=tx.bandwidth,
            rotation_time=rotation_time,
        ).receiver
        rx.noise_figure = 0.0
        rx.id = i
        pcl_sensors.append(
            PclSensor(
                id=i,
                transmitter=tx,
                receiver=rx,
                error_model=PclMeasurementModel(
                    min_bistatic_range_uncertainty=bistatic_range_uncertainty,
                    max_bistatic_range_uncertainty=bistatic_range_uncertainty,
                ),
            )
        )

    example_sensors = [pcl_sensors[0], pcl_sensors[11], pcl_sensors[15]]

    lat_min = 47.1497
    lat_max = 47.5351
    lat_res = 1 / 400.0
    lon_min = 8.0641
    lon_max = 8.8636
    lon_res = 1 / 400.0

    lats = np.arange(lat_min, lat_max, lat_res)
    lons = np.arange(lon_min, lon_max, lon_res)
    alts = np.array([1000])

    grid = LatLonHeightGrid(
        lat_start=lats[0],
        lat_stop=lats[-1],
        lat_res=lats[1] - lats[0] if len(lats) > 1 else 1.0,
        lon_start=lons[0],
        lon_stop=lons[-1],
        lon_res=lons[1] - lons[0] if len(lons) > 1 else 1.0,
        height_start=alts[0],
        height_stop=alts[-1],
        height_res=alts[1] - alts[0] if len(alts) > 1 else 1.0,
    )

    # Define the targets.
    # In track update region.
    p1 = Point(lat=47.2985, lon=8.1231, alt=1000.0)
    # In track init region.
    p2 = Point(lat=47.2985, lon=8.2667, alt=1000.0)
    # In track update region.
    p3 = Point(lat=47.2985, lon=8.4035, alt=1000.0)

    v = 300

    t1 = datetime.datetime(year=2026, month=4, day=20)
    dt1 = line_of_sight_distance(p1.lat, p1.lon, p1.alt, p2.lat, p2.lon, p2.alt) / v
    t2 = t1 + datetime.timedelta(seconds=dt1)
    dt2 = line_of_sight_distance(p2.lat, p2.lon, p2.alt, p3.lat, p3.lon, p3.alt) / v
    t3 = t2 + datetime.timedelta(seconds=dt2)

    trajectory = Trajectory(
        target_id=0,
        target_sidc=SIDC_RED_FIXED_WING,
        times=[t1, t2, t3],
        lats=[p1.lat, p2.lat, p3.lat],
        lons=[p1.lon, p2.lon, p3.lon],
        alts=[p1.alt, p2.alt, p3.alt],
        vxs=[0.0, 0.0, 0.0],
        vys=[0.0, 0.0, 0.0],
        vzs=[0.0, 0.0, 0.0],
        cross_section_model=ConstantRcsModel(rcs=rcs),
    )

    return (
        example_sensors,
        [trajectory],
        grid,
    )


def sample_position(
    rng: np.random.Generator,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    alt_min: float = 0.0,
    alt_max: float = 10_000.0,
    alt_magl: bool = True,
) -> Point:
    """
    Parameters
    ----------
    rng: np.random.Generator
        (Pseudo) random number generator
    lat_min: float, default -90.0
        Minimum latitude to sample [°]
    lat_max: float, default 90.0
        Maximum latitude to sample [°]
    lon_min: float, default -90.0
        Minimum longitude to sample [°]
    lon_max: float, default 90.0
        Maximum longitude to sample [°]
    alt_min: float, default 0.0
        Minimum altitude to sample [m]
    alt_max: float, default 10_000.0
        Maximum altitude to sample [m]
    alt_magl: float, default True
        Whether to interpret the altitude bounds as meters above ground level
        (``True``) or meters above sea level (``False``)
    """
    lat = rng.uniform(lat_min, lat_max)
    lon = rng.uniform(lon_min, lon_max)
    alt = rng.uniform(alt_min, alt_max)
    if alt_magl:
        alt += elevationAt(lat, lon)
    return Point(lat=lat, lon=lon, alt=alt)


def build_single_target_from_Bodensee(
    speed: float = 300.0,
    rcs: float = 1.0,
    alt: float = 1000.0,
) -> Trajectory:
    t_start = datetime.datetime(year=2026, month=4, day=29)
    p_start = Point(lat=47.9395, lon=9.1240, alt=alt)
    p_center = Point(lat=47.1501, lon=9.8984, alt=alt)

    maneuver = ConstantSpeedCurveManeuver(
        center_point=p_center,
        speed=speed,
        angular_arclength=np.deg2rad(-110),
    )
    times, points = maneuver.get_waypoints(t_start, p_start)
    timestamps = [t.timestamp() for t in times]
    points_ecef = np.array(
        [CoordinateTransformations.geodetic_to_cartesian(*p.as_tuple()) for p in points]
    )
    f_ecef = CubicSpline(timestamps, points_ecef, extrapolate=True)
    v_ecef = f_ecef.derivative()
    velocities = v_ecef(timestamps)

    return Trajectory(
        target_id=0,
        target_sidc=SIDC_RED_FIXED_WING,
        times=times,
        lats=[p.lat for p in points],
        lons=[p.lon for p in points],
        alts=[p.alt for p in points],
        vxs=[v[0] for v in velocities],
        vys=[v[1] for v in velocities],
        vzs=[v[2] for v in velocities],
        cross_section_model=ConstantRcsModel(rcs=rcs),
    )


def build_fighter_jet_radar(
    id: int,
    tx_id: int,
    rx_id: int,
    frequency: float = 9.0 * 1e3,
    antenna_diameter: float = 1.0,
    transmitter_bandwidth: float = 1.0,
    pulse_width: float = 300.0,
) -> MonostaticSensor:
    """
    Parameters
    ----------
    id: int
        ID of the sensor
    tx_id: int
        ID of the transmitter
    rx_id: int
        ID of the receiver
    frequency: float, default 9000
        Signal frequency [MHz]
    antenna_diameter: float, default 1.0
        Diameter of the antenna [m]
    transmitter_bandwidth: float, default 500
        Signal bandwidth [MHz]
    pulse_width: float, default 1.0
        Signal pulse width [us]
    """
    p = Point(lat=0, lon=0, alt=0)
    wavelength = frequency_to_wavelength(frequency)  # m
    antenna_gain = calculate_antenna_gain(antenna_diameter, wavelength)
    tx = Transmitter(
        id=tx_id,
        point=p,
        power=3_000,
        erp=3_000 * from_dB(antenna_gain),
        antenna_height=0.0,
        antenna_diameter=antenna_diameter,
        antenna_gain=antenna_gain,
        frequency=frequency,
        pulse_width=pulse_width,
        polarization=Polarization.VERTICAL,
        bandwidth=transmitter_bandwidth,
    )
    rx = Receiver(
        id=rx_id,
        point=p,
        antenna_height=0.0,
        diameter=antenna_diameter,
        cpi_pulses=10,
        pfa=1e-6,
        min_elevation=-90.0,
        max_elevation=90.0,
        rotation_time=1,
        bandwidth=1 / pulse_width,
        gain=antenna_gain,
        losses=3.0,
    )

    return MonostaticSensor(
        id=id,
        transmitter=tx,
        receiver=rx,
        error_model=MonostaticRadarMeasurementModel(),
    )


def build_flores_monostatic_radar(
    point: Point,
    sensor_id: int,
    rx_id: int,
    tx_id: int,
) -> MonostaticSensor:
    # Sources:
    # [Thales] https://www.radartutorial.eu/19.kartei/02.surv/pubs/Master_M.pdf
    # [Wiki] https://fr.wikipedia.org/wiki/Ground_Master_200
    # [Fandom] https://schweiz.fandom.com/de/wiki/FLORAKO
    frequency = 3 * 1e3  # 3 GHz, S-band [Thales]
    wavelength = frequency_to_wavelength(frequency)
    # Careful: [Thales] states 400MHz operating bandwidth, which is NOT the signal bandwidth!
    bandwidth = 5  # MHz
    antenna_diameter = 2  # m; guess - actually a flat antenna panel
    antenna_height = 8  # m [Wiki]
    power = 100_000  # W; guess
    gain = calculate_antenna_gain(antenna_diameter, wavelength)
    pulse_width = 10  # us; guess
    rotation_time = 4  # s; [Fandom]
    tx = Transmitter(
        id=tx_id,
        point=point,
        power=power,
        erp=power * from_dB(gain),
        antenna_height=antenna_height,
        antenna_diameter=antenna_diameter,
        antenna_gain=gain,
        frequency=frequency,
        pulse_width=pulse_width,
        polarization=Polarization.VERTICAL,
        bandwidth=bandwidth,
    )
    rx = Receiver(
        id=rx_id,
        point=point,
        antenna_height=antenna_height,
        diameter=antenna_diameter,
        cpi_pulses=10,
        pfa=1e-6,
        min_elevation=-90.0,
        max_elevation=90.0,
        rotation_time=rotation_time,
        bandwidth=1 / pulse_width,
        gain=calculate_antenna_gain(antenna_diameter, wavelength),
        losses=0.0,
    )

    error_model = MonostaticRadarMeasurementModel(
        min_range_uncertainty=30,  # m; [Thales]
        max_range_uncertainty=10_000,  # m; avoid infinity for serialization to JSON
        min_angular_uncertainty=np.deg2rad(0.3),  # rad; [Thales]
    )

    return MonostaticSensor(
        id=sensor_id,
        transmitter=tx,
        receiver=rx,
        error_model=error_model,
    )


def build_pcl_receiver(
    rx_id: int,
    point: Point,
    antenna_height: float = 8.0,
    antenna_diameter: float = 2.0,
) -> Receiver:
    """
    Parameters
    ----------
    rx_id: int
        Unique ID of the receiver
    point: Point
        Position of the receiver
    antenna_height: float, default 8.0
        Antenna height [m]
    antenna_diameter: float, default 2.0
        Antenna diameter [m]
    """
    # Default values are inspired by Hensoldt TwinVis:
    # https://www.hensoldt.net/products/twinvis-passive-radar-surveillance-of-noiseless-objects
    return Receiver(
        id=rx_id,
        point=point,
        antenna_height=antenna_height,
        diameter=antenna_diameter,
        cpi_pulses=1,
        pfa=-1.0,  # not used for PCL
        min_elevation=0.0,  # currently not used
        max_elevation=np.pi / 2,  # currently not used
        rotation_time=1.0,
        # FM modulating frequency <= 53 kHz, peak deviation 75kHz. Apply Carson's rule.
        # Source:
        # https://en.wikipedia.org/wiki/Carson_bandwidth_rule
        bandwidth=0.256,  # MHz
    )
