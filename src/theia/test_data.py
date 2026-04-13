import copy
from typing import Literal

import numpy as np
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.terrain import elevationAt
from theia.types import (
    ConstantRcsModel,
    MonostaticRadarMeasurementModel,
    Point,
    Polarization,
    Radar,
    Receiver,
    Situation,
    Target,
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
    ) -> Situation:
        radar_lat = POSITIONS_OF_INTEREST["Uetliberg"]["lat"]
        radar_lon = POSITIONS_OF_INTEREST["Uetliberg"]["lon"]
        point = Point(
            lat=radar_lat, lon=radar_lon, alt=elevationAt(radar_lat, radar_lon)
        )

        radar = get_uetliberg_radar(
            frequency=1000.0,
            diameter=2.0,
            bandwidth=100,
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

        return Situation(
            radars=[radar],
            targets=[target],
            transmitter_labels={0: "Tx"},
            target_labels={0: "Target (southbound)"},
        )


def get_uetliberg_radar(
    frequency: float = 3_000,
    power: float = 500_000,
    diameter: float = 4.0,
    bandwidth: float = 5.0,
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
        tx_bandwidth = bandwidth

    return Radar(
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
            bandwidth=bandwidth,
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
            bandwidth=bandwidth,
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
