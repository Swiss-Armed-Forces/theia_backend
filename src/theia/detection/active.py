import datetime
import math
from typing import Optional
import numpy as np
import scipy.constants as sc
from theia.config import ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD, RF_LOSS
from theia.coordinates import CoordinateTransformations
from theia.distance import line_of_sight_distance
from theia.doppler import calculate_doppler_shift
from theia.line_of_sight import has_line_of_sight
from theia.measurement import MonostaticMeasurementTransformations
from theia.snr import calculate_snr
from theia.types import (
    MonostaticRadarDetection,
    MonostaticRadarMeasurementModel,
    Radar,
    RcsModel,
    Target,
)
from theia.util import get_clear_sky_attenuation, marcum_q_function


def calculate_monostatic_detection(
    radar: Radar,
    target: Target,
    rng: np.random.Generator,
    distance_step: float = 30.0,
    doppler_shift_threshold_hz: float = ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD,
    rf_loss: float = RF_LOSS,
    error_model: Optional[MonostaticRadarMeasurementModel] = None,
) -> MonostaticRadarDetection | None:
    """
    Calculate probabilistic active radar detection.

    Parameters
    ----------
    radar: Radar
        Radar
    target: Target
        Target
    rng: np.random.Generator
        Random number generator to use during the detection (is modified)
    distance_step: float, default 30.0
        Distance stepping to be used for the line-of-sight test [m]
    doppler_shift_threshold_hz: float, default 5.0
        Threshold [Hz] for the Doppler shift detection criterion.
        Targets moving with less Doppler shift have zero detection probability.
    rf_loss: float, default 12.0
        RF system hardware loss [dB].
        TODO is this the transmitter leakage in the source below?
        Default value from chapter 2.12, p.80 Skolnik "Introduction to radar systems"
        TODO is this the same as the Receiver.losses property?
    error_model: Optional[MonostaticRadarMeasurementModel], default None
        Error model to use on the range, elevation and azimuth
    """
    snr_dB = calculate_monostatic_snr(
        radar,
        target,
        rcs_model=target.cross_section_model,
        distance_step=distance_step,
        doppler_shift_threshold_hz=doppler_shift_threshold_hz,
        rf_loss=rf_loss,
    )
    p = calculate_probability_of_detection(
        snr_dB,
        radar.receiver.pfa,
    )
    if rng.uniform(low=0, high=1) <= p:
        target_position_cartesian = CoordinateTransformations.geodetic_to_cartesian(
            target.lat,
            target.lon,
            target.alt,
        )
        elevation, azimuth, target_range = (
            MonostaticMeasurementTransformations.cartesian_to_elevation_azimuth_range(
                radar.receiver.point,
                target_position_cartesian,
            )
        )
        sigma_range = 0.0
        sigma_elevation = 0.0
        sigma_azimuth = 0.0
        if error_model is not None:
            sigma_range = error_model.calculate_range_uncertainty(radar, snr_dB)
            sigma_elevation = error_model.calculate_elevation_uncertainty(radar, snr_dB)
            sigma_azimuth = error_model.calculate_azimuth_uncertainty(radar, snr_dB)
            target_range += rng.normal(loc=0.0, scale=sigma_range)
            elevation += rng.normal(loc=0.0, scale=sigma_elevation)
            azimuth += rng.normal(loc=0.0, scale=sigma_azimuth)
        target_range = np.clip(target_range, a_min=0.0, a_max=np.inf)
        elevation = np.clip(elevation, a_min=-np.pi / 2.0, a_max=np.pi / 2)
        azimuth = np.clip(azimuth, a_min=0, a_max=2 * np.pi - 1e-6)
        return MonostaticRadarDetection(
            detection_id=-1,
            time=datetime.datetime.fromtimestamp(0),
            radar=radar,
            target=target,
            snr=snr_dB,
            target_range=target_range,
            elevation_angle=elevation,
            azimuth_angle=azimuth,
            sigma_target_range=sigma_range,
            sigma_elevation=sigma_elevation,
            sigma_azimuth=sigma_azimuth,
        )
    else:
        return None


def calculate_monostatic_snr(
    radar: Radar,
    target: Target,
    rcs_model: RcsModel,
    distance_step: float,
    doppler_shift_threshold_hz,
    rf_loss,
) -> float:
    """
    Calculate signal-to-noise ratio for a monostatic radar.

    Parameters
    ----------
    radar: Radar
        Radar
    target: Target
        Target
    rcs_model: RcsModel
        Model to be used to estimate the radar cross section
    distance_step: float
        Distance stepping to be used for the line-of-sight test [m]
    doppler_shift_threshold_hz: float
        Threshold [Hz] for the Doppler shift detection criterion.
        Targets moving with less Doppler shift have zero detection probability.
    rf_loss: float
        RF system hardware loss [dB].
        TODO is this the transmitter leakage in the source below?
        Default value from chapter 2.12, p.80 Skolnik "Introduction to radar systems"

    Returns
    -------
    snr: float
        Signal-to-noise ration [dB]
    """
    rad_lat = radar.transmitter.lat
    rad_lon = radar.transmitter.lon
    rad_height = radar.transmitter.alt

    tgt_lat = target.lat
    tgt_lon = target.lon
    tgt_height = target.alt

    dist = line_of_sight_distance(
        rad_lat,
        rad_lon,
        rad_height,
        tgt_lat,
        tgt_lon,
        tgt_height,
    )

    los_ok = has_line_of_sight(radar.transmitter.point, target.point, distance_step)

    if not los_ok:
        return 0.0
    else:
        doppler = calculate_doppler_shift(
            radar.receiver,
            target,
            radar.transmitter,
        )
        if abs(doppler) <= doppler_shift_threshold_hz:
            return 0.0

    wavelength = sc.speed_of_light / (radar.transmitter.frequency * 1e6)
    snr_dB = calculate_snr(
        wavelength=wavelength,
        antenna_gain_transmitter=radar.transmitter.antenna_gain,
        antenna_gain_receiver=radar.receiver.antenna_gain(radar.transmitter.frequency),
        radar_cross_section=rcs_model(
            transmitter=radar.transmitter,
            receiver=radar.receiver,
            target=target,
        ),
        distance_transmitter_target=dist,
        distance_receiver_target=dist,
        transmission_power=radar.transmitter.power,
        bandwidth=radar.receiver.bandwidth,
        cpi_pulses=radar.receiver.cpi_pulses,
        equivalent_temperature=radar.receiver.noise_temperature,
        L_t=rf_loss,
        L_a=get_clear_sky_attenuation(radar.transmitter.frequency) * 2 * dist / 1000.0,
        # TODO: Should we include these factors?
        polarization_factor=0.0,
        pattern_propagation_factor_receiver=0.0,
        pattern_propagation_factor_transmitter=0.0,
    )
    return snr_dB


def calculate_probability_of_detection(snr: float, pfa: float) -> float:
    """
    Calculate the probability of detection.

    Parameters
    ----------
    snr: float
        Signal-to-noise ratio [dB]
    pfa: float
        probability of false alarm in (0, 1]

    Notes
    -----
    This function implements Equ. (22), (23) in Brennan & Reed, 1973.

    References
    ----------
    Brennan, L. E., & Reed, L. S. (1973). Theory of Adaptive Radar.
    IEEE Transactions on Aerospace and Electronic Systems, AES-9(2), 237-252.
    https://doi.org/10.1109/taes.1973.309792
    """
    alpha = pow(10, ((snr + 3) / 20))
    # Avoid blowup in Bessel function.
    if alpha > 30:
        return 1.0
    beta = math.sqrt(-2 * (np.log(pfa)))
    return marcum_q_function(alpha, beta)
