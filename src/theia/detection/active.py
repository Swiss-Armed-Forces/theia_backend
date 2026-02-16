import datetime
import math
import numpy as np
import scipy.constants as sc
from theia.config import ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD, RF_LOSS
from theia.coordinates import calculate_azimuth_angle, calculate_elevation_angle
from theia.distance import line_of_sight_distance
from theia.doppler import calculate_doppler_shift
from theia.line_of_sight import has_line_of_sight
from theia.snr import calculate_snr
from theia.types import ActiveRadarDetection, Radar, RcsModel, Target
from theia.util import get_clear_sky_attenuation, marcum_q_function


def calculate_monostatic_detection(
    radar: Radar,
    target: Target,
    rng: np.random.Generator,
    distance_step: float = 30.0,
    doppler_shift_threshold_hz: float = ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD,
    rf_loss: float = RF_LOSS,
) -> ActiveRadarDetection | None:
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
    """
    p = get_rad_pd(
        radar,
        target,
        rcs_model=target.cross_section_model,
        distance_step=distance_step,
        doppler_shift_threshold_hz=doppler_shift_threshold_hz,
        rf_loss=rf_loss,
    )
    if rng.uniform(low=0, high=1) <= p:
        return ActiveRadarDetection(
            detection_id=-1,
            time=datetime.datetime.fromtimestamp(0),
            radar=radar,
            target=target,
            target_range=line_of_sight_distance(
                *radar.transmitter.point.as_tuple(),
                *target.point.as_tuple(),
            ),
            elevation_angle=calculate_elevation_angle(
                radar.transmitter.point,
                target.point,
            ),
            azimuth_angle=calculate_azimuth_angle(
                radar.transmitter.point,
                target.point,
            ),
        )
    else:
        return None


def get_rad_pd(
    radar: Radar,
    target: Target,
    rcs_model: RcsModel,
    distance_step: float,
    doppler_shift_threshold_hz,
    rf_loss,
) -> float:
    """
    Calculate probability of detection for a monostatic radar.

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
        print("No LOS")
        return 0.0
    else:
        doppler = calculate_doppler_shift(
            radar.receiver,
            target,
            radar.transmitter,
        )
        if doppler <= doppler_shift_threshold_hz:
            print("Doppler fail")
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
        bandwidth=radar.transmitter.bandwidth,
        cpi_pulses=radar.receiver.cpi_pulses,
        equivalent_temperature=radar.receiver.noise_temperature,
        L_t=rf_loss,
        L_a=get_clear_sky_attenuation(radar.transmitter.frequency) * 2 * dist / 1000.0,
        # TODO: Should we include these factors?
        polarization_factor=0.0,
        pattern_propagation_factor_receiver=0.0,
        pattern_propagation_factor_transmitter=0.0,
    )
    print(f"SNR = {snr_dB:.2f}dB")

    # avoid segmentation fault in the besseli function for high snr values
    return (
        1.0
        if snr_dB > 30
        else calculate_probability_of_detection(
            snr_dB,
            radar.receiver.pfa,
        )
    )


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
    beta = math.sqrt(-2 * (np.log(pfa)))
    return marcum_q_function(alpha, beta)
