import datetime
import math

import numpy as np
import scipy.constants as sc

from theia.coordinates import get_azimuth_between_locs
from theia.distance import (
    get_2d_distance_between_locs_heights,
    get_bistatic_range,
    get_elev_angle,
)
from theia.doppler import calculate_bistatic_doppler
from theia.line_of_sight import has_line_of_sight
from theia.types import PassiveRadarDetection, Point, Radar, Target
from theia.util import to_dB


def calculate_bistatic_detection(
    rx: Radar,
    tx: Radar,
    tgt: Target,
    snr_thresh: float = 15.0,
    doppler_thresh: float = 2.0,
    delay_thresh: float = 1.0,  # us
) -> PassiveRadarDetection | None:
    """sets PCL live detections

    Parameters
    ----------
    rx: Radar
        Receiver.
    tx: Radar:
        Transmitter.
    tgt: Target
        Target.
    snr_thresh: float, default 15.0
        Signal-to-noise threshold [dB]
    doppler_thresh: float, default 2.0
        Doppler threshold [Hz]

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
    doppler = calculate_bistatic_doppler(rx, tgt, tx)  # [Hz]

    if abs(doppler) < doppler_thresh:
        return None

    # dist_delay_limit [m], delay_thresh [us], baseline_range in [km] * 1000 in [m]
    dist_delay_limit = delay_thresh * sc.speed_of_light / 1e6 + (baseline_range * 1000)

    snr = calculate_snr(tx, rx, tgt.point, dist_delay_limit)
    min_detectable_rcs = calculate_minimum_detectable_rcs(snr, snr_thresh)

    print(snr)
    print(min_detectable_rcs)

    assert min_detectable_rcs > 0.

    # Doppler shift was good enough if we reached this far, see above.
    # So just check the rcs_thresholds.
    if min_detectable_rcs <= tgt.cross_section:
        return PassiveRadarDetection(
            detection_id=-1,
            time=datetime.datetime.fromtimestamp(0),
            transmitter=tx,
            receiver=rx,
            target=tgt,
            bistatic_range=bistatic_range_km * 1000.,
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


def get_clear_sky_attenuation(transmitter_freq: float) -> float:
    """
    Calculate clear sky atmospheric one-way attenuation [dB/km] for Radar Windows.

    Parameters
    ----------
    transmitter_freq: float
        Transmitter frequency [MHz].

    Returns
    -------
    float
        Clear sky atmospheric one-way attenuation [dB/km]

    Notes
    -----
    Clear Sky weather values from Barton book: 'Modern Radar System Analysis'.
    """

    freq_mhz = [200, 500, 1000, 10000]
    atten = [0.00075, 0.003, 0.0055, 0.012]
    # one-way attenuation: dB/km therefore division by 2
    # ???
    atten_db = max(np.interp(transmitter_freq, freq_mhz, atten), 0)

    return atten_db


def calculate_snr(
    Tx: Radar,
    Rx: Radar,
    point_of_interest: Point,
    dist_delay_limit: float,
):
    r"""
    Calculate the signal-to-noise ratio (SNR) [dB].

    Parameters
    ----------
    Rx: Radar
        Receiver.
    Tx: Radar
        Transmitter.
    point_of_interest: Point
        Position at which we'd like to query the minimum detectable RCS.
    dist_delay_limit: float
        Distance limit for the delay. The bistatic regime is left if the delay
        distance is below this limit, which will raise a ``ValueError``.
    snr_threshold: float
        Signal-to-noise-threshold [dB], i. e. the minimum SNR to have for a detection.

    Raises
    ------
    ValueError
        If the setup is not in the bistatic regime (e. g. the forward scattering)

    Notes
    -----
    The SNR is calculated according to the radar equation

    .. math::

        SNR_{\setminus \sigma_B} := \frac{SNR}{\sigma_B} = \frac{P_{T, EIRP} G_R G_p \lambda^2 F_T^2 F_R^2}{(4 \pi)^3 k_B T_S B_n L_T L_R} \frac{1}{R_T^2 R_R^2},

    where :math:`P{T, EIRP}` is the transmitter's EIRP,
    :math:`G_R, G_P` are the receiving antenna and processing gain,
    :math:`\lambda` is the signal wavelength, :math:`k_T T_S B_n` is the effective
    input noise power. The quantities :math:`F_T, F_R` are angle- and frequency-dependent pattern
    propagation factors.

    The quantities :math`R_T, R_R` represent the distance
    between transmitter and point of interest as well as receiver and point of interest.

    Finally, the quantities :math:`L_T, L_R` represent other losses,
    including atmospheric absorption and line-feed losses between transmitter output
    and transmitter antenna as well as between receiving antenna output to receiver input.
    """
    # Check whether the delay threshold is kept.
    r_r = (
        get_2d_distance_between_locs_heights(
            Rx.lat,
            Rx.lon,
            Rx.alt + Rx.antenna_height,
            point_of_interest.lat,
            point_of_interest.lon,
            point_of_interest.alt,
        )
        * 1000.0
    )  # distance in meters
    print(f"Rx.lat = {Rx.lat}")
    print(f"Rx.lon = {Rx.lon}")
    print(f"Rx.alt + Rx.antenna_height = {Rx.alt + Rx.antenna_height}")
    print(f"point_of_interest.lat = {point_of_interest.lat}")
    print(f"point_of_interest.lon = {point_of_interest.lon}")
    print(f"point_of_interest.alt = {point_of_interest.alt}")

    r_t = (
        get_2d_distance_between_locs_heights(
            Tx.lat,
            Tx.lon,
            Tx.alt + Tx.antenna_height,
            point_of_interest.lat,
            point_of_interest.lon,
            point_of_interest.alt,
        )
        * 1000.0
    )  # distance in meters

    if r_r + r_t < dist_delay_limit:  # checks if delay threshold is valid
        raise ValueError("Delay is too small; we are in the forward scattering regime")

    # Evaluate attenuation for the angles of gaze.
    theta_t_bearing = get_azimuth_between_locs(
        Tx.lat,
        Tx.lon,
        point_of_interest.lat,
        point_of_interest.lon,
    )
    theta_t_vert = get_elev_angle(
        point_of_interest.alt,
        Tx.alt + Tx.antenna_height,
        r_t,
    )
    theta_r_bearing = get_azimuth_between_locs(
        Rx.lat,
        Rx.lon,
        point_of_interest.lat,
        point_of_interest.lon,
    )
    theta_r_vert = get_elev_angle(
        point_of_interest.alt,
        Rx.alt + Rx.antenna_height,
        r_r,
    )

    tx_horiz_att = Tx.horizontal_attenuation(theta_t_bearing)
    rx_horiz_att = Rx.horizontal_attenuation(theta_r_bearing)
    tx_vert_att = Tx.vertical_attenuation(theta_t_vert)
    rx_vert_att = Rx.vertical_attenuation(theta_r_vert)

    beam_shape_loss_dB = rx_horiz_att + tx_horiz_att + tx_vert_att + rx_vert_att
    eirp_dBW = to_dB(Tx.erp) + 2.15
    atmospheric_loss_dB = get_clear_sky_attenuation(Tx.frequency) * (
        (r_t + r_r) / 1000.0
    )
    free_space_loss_dB = 20 * np.log10(r_t * r_r)
    rx_thermal_noise_loss_dB = 10 * np.log10(
        sc.Boltzmann * Rx.noise_temperature * Rx.bandwidth * 1e6
    )
    wavelength_squared_dB = 20 * np.log10(sc.speed_of_light / (Tx.frequency * 1e6))

    print(f"r_r = {r_r}")
    print(f"r_t = {r_t}")
    print(f"theta_t_vert = {theta_t_vert}")
    print(f"theta_r_vert = {theta_r_vert}")
    print(f"tx_horiz_att = {tx_horiz_att}")
    print(f"rx_horiz_att = {rx_horiz_att}")
    print(f"tx_vert_att = {tx_vert_att}")
    print(f"rx_vert_att = {rx_vert_att}")
    print("[" + ','.join(np.array([eirp_dBW, Rx.gain, wavelength_squared_dB, Tx.processing_gain, Rx.losses, atmospheric_loss_dB, rx_thermal_noise_loss_dB, free_space_loss_dB,beam_shape_loss_dB]).astype(str)) + "]")

    return (
        eirp_dBW
        + Rx.gain
        # + wavelength_squared_dB
        + Tx.processing_gain
        - abs(Rx.losses)
        # - atmospheric_loss_dB
        - rx_thermal_noise_loss_dB
        # - free_space_loss_dB
        + beam_shape_loss_dB
    )
