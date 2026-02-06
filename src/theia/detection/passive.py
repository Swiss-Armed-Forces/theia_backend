import datetime

import scipy.constants as sc

from theia.config import (
    DELAY_THRESHOLD_PCL,
    DOPPLER_SHIFT_THRESHOLD_PCL,
    SNR_THRESHOLD_PCL,
)
from theia.coordinates import calculate_azimuth_angle, calculate_elevation_angle
from theia.distance import (
    get_2d_distance_between_locs_heights,
    get_bistatic_range,
)
from theia.doppler import calculate_doppler_shift
from theia.snr import calculate_snr
from theia.types import (
    PassiveRadarDetection,
    Point,
    RcsModel,
    Receiver,
    Target,
    Transmitter,
)
from theia.util import get_clear_sky_attenuation


def calculate_bistatic_detection(
    rx: Receiver,
    tx: Transmitter,
    tgt: Target,
    rcs_model: RcsModel,
    snr_thresh: float = SNR_THRESHOLD_PCL,
    doppler_thresh: float = DOPPLER_SHIFT_THRESHOLD_PCL,
    delay_thresh: float = DELAY_THRESHOLD_PCL,
) -> PassiveRadarDetection | None:
    """sets PCL live detections

    Parameters
    ----------
    rx: Receiver
        Receiver.
    tx: Transmitter
        Transmitter.
    tgt: Target
        Target.
    rcs_model: RcsModel
        Model to be used to estimate the radar cross section
    snr_thresh: float, default theia.config.SNR_THRESHOLD_PCL
        Signal-to-noise threshold [dB]
    doppler_thresh: float, default theia.config.DOPPLER_SHIFT_THRESHOLD_PCL
        Doppler threshold [Hz]
    delay_thresh: float, default theia.config.DELAY_THRESHOLD_PCL
        Delay threshold [us]. If the delay
        (signal propagation time deviation from straight line between Tx and Rx)
        is smaller than this value, the geometry is considered to fall into the
        forward scattering domain and a ValueError is raised.

    Raises
    ------
    ValueError
        If the setup is not in the bistatic regime (e. g. the forward scattering)

    Returns
    -------
    detection: PassiveRadarDetection | None
        the detection that was made or None if no detection takes place.
    """
    assert tgt.alt > 0

    # Get bistatic range.
    tx_pos = [tx.lat, tx.lon, tx.alt + tx.antenna_height]
    rx_pos = [rx.lat, rx.lon, rx.alt + rx.antenna_height]
    tgt_pos = [tgt.lat, tgt.lon, tgt.alt]
    (bistatic_range_km, tgt_rx_range, tgt_tx_range, baseline_range) = (
        get_bistatic_range(tx_pos, rx_pos, tgt_pos)
    )

    # Now check bistatic Doppler and return if Doppler shift too low.
    doppler = calculate_doppler_shift(rx, tgt, tx)  # [Hz]

    if abs(doppler) < doppler_thresh:
        return None

    # Check whether we are in the bistatic regime. Otherwise raise an exception.
    # dist_delay_limit [m], delay_thresh [us], baseline_range in [km] * 1000 in [m]
    dist_delay_limit = delay_thresh * sc.speed_of_light / 1e6 + (baseline_range * 1000)

    r_r = (
        get_2d_distance_between_locs_heights(
            rx.lat,
            rx.lon,
            rx.alt + rx.antenna_height,
            tgt.lat,
            tgt.lon,
            tgt.alt,
        )
        * 1000.0
    )  # distance in meters

    r_t = (
        get_2d_distance_between_locs_heights(
            tx.lat,
            tx.lon,
            tx.alt + tx.antenna_height,
            tgt.lat,
            tgt.lon,
            tgt.alt,
        )
        * 1000.0
    )  # distance in meters

    if r_r + r_t < dist_delay_limit:  # checks if delay threshold is valid
        raise ValueError("Delay is too small; we are in the forward scattering regime")

    snr = calculate_snr(
        wavelength=sc.speed_of_light / (tx.frequency * 1e6),
        antenna_gain_transmitter=tx.antenna_gain,
        antenna_gain_receiver=rx.antenna_gain(tx.frequency),
        radar_cross_section=1.0,
        distance_transmitter_target=r_t,
        distance_receiver_target=r_r,
        transmission_power=tx.power,
        bandwidth=rx.bandwidth,
        cpi_pulses=1,
        equivalent_temperature=rx.noise_temperature,
        L_t=0.0,
        L_a=get_clear_sky_attenuation(tx.frequency) * (r_r + r_t) / 1000.0,
        polarization_factor=0,
        pattern_propagation_factor_transmitter=calculate_antenna_pattern(tx, tgt.point),
        pattern_propagation_factor_receiver=calculate_antenna_pattern(rx, tgt.point),
    )
    min_detectable_rcs = calculate_minimum_detectable_rcs(snr, snr_thresh)

    assert min_detectable_rcs > 0.0

    # Doppler shift was good enough if we reached this far, see above.
    # So just check the rcs_thresholds.
    if min_detectable_rcs <= rcs_model(
        transmitter=tx,
        receiver=rx,
        target=tgt,
    ):
        return PassiveRadarDetection(
            detection_id=-1,
            time=datetime.datetime.fromtimestamp(0),
            transmitter=tx,
            receiver=rx,
            target=tgt,
            bistatic_range=bistatic_range_km * 1000.0,
            doppler_shift=doppler,
        )
    else:
        return None


def calculate_minimum_detectable_rcs(
    snr_over_rcs: float,
    snr_threshold: float,
) -> float:
    r"""
    Calculate the minimum radar cross section (RCS) that can be detected for
    the given geometry according to an SNR threshold test.

    Parameters
    ----------
    snr: float
        Signal-to-noise ratio [dB] divided by the radar cross section [m^2]
        for the given geometry.
    snr_threshold: float
        Minimum detectable SNR threshold [dB].
    
    Returns
    -------
    min_detectable_rcs: float
        Minimum detectable radar cross section [m^2] corresponding to
        the given SNR and SNR threshold.

    Notes
    -----
    The principle is described in the unpublished master thesis with the
    title "Passive Radar - From Quality Criteria to Coverage Optimization".
    Since the bistatic RCS :math:`\sigma_B` is unknown, the original expression
    for the SNR is reformulated and only the term :math:`SNR / \sigma_B`,
    which can be computed, is used.

    The derivation of the minimum detectable RCS is straightforward

    .. math::

       \begin{aligned}
       SNR &\geq SNR_{threshold}\\
       \frac{SNR}{\sigma_B} &\geq \frac{SNR_{threshold}}{\sigma_B}\\
       \sigma_B &\geq \frac{SNR_{threshold}}{SNR_{\setminus \sigma_B}} =: \sigma_{B, min}
       \end{aligned}.

    """
    rcs = 10.0 ** ((snr_threshold - snr_over_rcs) / 10.0)
    return rcs


def calculate_antenna_pattern(
    transmitter_or_receiver: Transmitter | Receiver,
    point_of_interest: Point,
) -> float:
    """
    Calculate antenna pattern attenuation [dB] for the given geometry.
    """
    p = Point(
        lat=transmitter_or_receiver.lat,
        lon=transmitter_or_receiver.lon,
        alt=transmitter_or_receiver.alt + transmitter_or_receiver.antenna_height,
    )
    theta_bearing = calculate_azimuth_angle(
        p_observer=p,
        p_target=point_of_interest,
    )
    theta_vert = calculate_elevation_angle(
        p_observer=p,
        p_target=point_of_interest,
    )

    horiz_att = transmitter_or_receiver.horizontal_attenuation(theta_bearing)
    vert_att = transmitter_or_receiver.vertical_attenuation(theta_vert)

    return horiz_att + vert_att
