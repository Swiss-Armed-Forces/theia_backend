import copy
import datetime
from typing import Literal

import numpy as np
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.data_loading import load_bakom_ukw_transmitters
from theia.distance import line_of_sight_distance
from theia.grids import LatLonHeightGrid
from theia.terrain import elevationAt
from theia.types import (
    ConstantRcsModel,
    MonostaticRadarMeasurementModel,
    Point,
    Polarization,
    Sensor,
    Receiver,
    Target,
    Trajectory,
    Transmitter,
    calculate_antenna_gain,
)
from theia.util import frequency_to_wavelength


class TestSituationLoader:
    @staticmethod
    def load_zuerich_single_target(
        speed: float,
        move_direction: Literal["north"]
        | Literal["east"]
        | Literal["south"]
        | Literal["west"] = "south",
    ) -> tuple[list[Sensor], list[Target]]:
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
            point=p_target,
            cross_section_model=ConstantRcsModel(rcs=2.0),
            velocity=v,
        )

        return [radar], [target]


def get_uetliberg_radar(
    frequency: float = 3_000,
    power: float = 500_000,
    diameter: float = 4.0,
    rx_bandwidth: float = 5.0,
    tx_bandwidth: float | None = None,
    integration_time: float = 0.1,
    min_range_uncertainty: float = 100,
    max_range_uncertainty: float = np.inf,
    min_angular_uncertainty: float = np.deg2rad(1),
    max_angular_uncertainty: float = np.deg2rad(360),
):
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

    return Sensor(
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
            rotation_time=10.0,
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


def load_pcl_example(rcs: float=1.0) -> tuple[list[Sensor], list[Trajectory], LatLonHeightGrid]:
    # Define the sensors.
    rx = get_uetliberg_radar(
        min_range_uncertainty=0.0,
        max_range_uncertainty=0.0,
        min_angular_uncertainty=0.0,
        max_angular_uncertainty=0.0,
    ).receiver

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

    pcl_sensors: list[Sensor] = []
    for i, tx in enumerate(transmitters):
        rx = get_uetliberg_radar(
            rx_bandwidth=tx.bandwidth, tx_bandwidth=tx.bandwidth
        ).receiver
        rx.id = i
        pcl_sensors.append(
            Sensor(
                id=i,
                transmitter=tx,
                receiver=rx,
                error_model=MonostaticRadarMeasurementModel(),
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
    p2 = Point(lat=47.2985, lon=8.2796, alt=1000.0)
    # In track update region.
    p3 = Point(lat=47.2985, lon=8.4035, alt=1000.0)

    v = 300

    t0 = datetime.datetime(year=2026, month=4, day=20)
    dt1 = line_of_sight_distance(p1.lat, p1.lon, p1.alt, p2.lat, p2.lon, p2.alt) / v
    t1 = t0 + datetime.timedelta(seconds=dt1)
    dt2 = line_of_sight_distance(p2.lat, p2.lon, p2.alt, p3.lat, p3.lon, p3.alt) / v
    t2 = t1 + datetime.timedelta(seconds=dt2)

    trajectory = Trajectory(
        target_id=0,
        times=[t0, t1, t2],
        lats=[p1.lat, p2.lat, p3.lat],
        lons=[p1.lon, p2.lon, p3.lon],
        alts=[p1.alt, p2.alt, p3.alt],
        vxs=[0.0, 0.0, 0.0],
        vys=[0.0, 0.0, 0.0],
        vzs=[0.0, 0.0, 0.0],
        cross_section_model=ConstantRcsModel(rcs=rcs),
    )

    return example_sensors, [trajectory], grid,


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
