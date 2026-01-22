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
from theia.types import Point, Radar, Target


def calculate_bistatic_detection(
    rx: Radar,
    tx: Radar,
    tgt: Target,
    radioprop_enabled: bool = False,
    snr_thresh: float = 15.0,
    doppler_thresh: float = 2.0,
    minimal_rcs_threshold: float = 150,
    min_bistatic_rcs_online_detection: float = 10.0,
    max_tgt_rx_range: float = 180.0,
    max_tgt_tx_range: float = 180.0,
) -> tuple[float, float]:
    """sets PCL live detections

    Parameters
    ----------
    rx: Radar
        Receiver.
    tx: Radar:
        Transmitter.
    tgt: Target
        Target.
    radioprop_enabled: bool = False
        Whether to use a propagation model. Currently, only False is implemented.
    snr_thresh: float, default 15.0
        Signal-to-noise threshold [dB]
    doppler_thresh: float, default 2.0
        Doppler threshold [dB]
    minimal_rcs_threshold: float, default 150.
        Minimum detectable radar cross section [m^2]
    min_bistatic_rcs_online_detection: float, default 10.0
        minimal bistatic rcs for online detection (i.e. the target is only
        detected if its bistatic RCS is above this value)
    max_target_rx_range: float, default 180.
        maximum range [km] considered between target and Rx for detection
    max_target_tx_range: float, default 180.
        maximum range [km] considered between target and Tx for detection

    Returns
    -------
    bistatic_range: float
        Bistatic range [km], i. e. Distance(Tx - target - Rx) - distance(Tx - Rx).
        np.nan if no detection was made.
    doppler: float
        Doppler shift [Hz] or np.nan if no detection was made.
    """
    ##################################################################
    ### setting just_los to 0 will invoke the usage of propagation loss computation in SPLAT
    ### setting just_los to 1 makes the SPLAT computation easier, ie without propagation losses
    just_los = not radioprop_enabled
    assert tgt.alt > 0

    # Get bistatic range.
    tx_pos = [tx.lat, tx.lon, tx.alt + tx.antenna_height]
    rx_pos = [rx.lat, rx.lon, rx.alt + rx.antenna_height]
    tgt_pos = [tgt.lat, tgt.lon, tgt.alt]
    (bistatic_range_km, tgt_rx_range, tgt_tx_range, baseline_range) = (
        get_bistatic_range(tx_pos, rx_pos, tgt_pos)
    )

    # We constraint the distance to ease computation (rule of thumb max dist for PCL).
    if (tgt_rx_range > max_tgt_rx_range) or (tgt_tx_range > max_tgt_tx_range):
        return np.nan, np.nan

    # Now check bistatic Doppler and return if Doppler shift too low.
    doppler = calculate_bistatic_doppler(rx, tgt, tx)  # [Hz]

    if abs(doppler) < doppler_thresh:
        return np.nan, np.nan

    rcs, snr = _calculate_snr(
        tx,
        rx,
        tgt,
        baseline_range,
        snr_thresh,
        just_los=just_los,
    )

    print(snr)

    # This means that the target with set bistatic RCS cannot be detected
    # with SNR above SNR_threshold.
    if rcs > min_bistatic_rcs_online_detection:
        return np.nan, np.nan
    # The snr computation could not be completed.
    if rcs < 0:
        return np.nan, np.nan

    # Doppler shift was good enough if we reached this far, see above.
    # So just check the snr and the rcs_thresholds
    if (snr > snr_thresh) and (rcs < minimal_rcs_threshold):
        return bistatic_range_km, doppler
    else:
        return np.nan, np.nan


def _calculate_snr(
    tx: Radar,
    rx: Radar,
    tgt: Target,
    baseline_range: float,
    snr_thresh: float,
    just_los: bool = True,
    delay_thresh: float = 1.0,  # us
    static_rcs: float = 0.0,
    spatial_res: float = 30.0,
) -> tuple[float, float]:
    """
    Calculate the signal-to-noise ratio for the given transmitter - target - receiver path.

    Parameters
    ----------
    tx: Radar
        Transmitter
    rx: Radar
        Receiver
    tgt: Target
        Target
    baseline_range: float
        Line-of-sight distance(between Rx - Tx) [km]
    snr_thresh: float
        Signal-to-noise threshold [dB]
    just_los: bool, default True
        Whether to use simple line-of-sight (True) or a propagation model (False).
        Currently, only the value True is supported.
    delay_thresh: float, default 1.0
        Delay threshold [us]
    static_rcs: float, default 0.0
        Static radar cross section [m^2]
    spatial_res: float, default 30.0
        Spatial resolution for the line-of-sight check

    Returns
    -------
    rcs: float
        Minimum detectable bistatic radar cross section.
    snr: float
        Signal-to-noise ratio
    """
    if just_los:
        tgt_rx_los = has_line_of_sight(
            Point(
                lat=rx.lat,
                lon=rx.lon,
                alt=rx.alt + rx.antenna_height,
            ),
            tgt.point,
            spatial_res,
        )
        tgt_tx_los = has_line_of_sight(
            Point(
                lat=tx.lat,
                lon=tx.lon,
                alt=tx.alt + tx.antenna_height,
            ),
            tgt.point,
            spatial_res,
        )
        propagation_loss = 0.0
    else:
        raise NotImplementedError("Propagation is not implemented yet!")

    # if not tgt_rx_los or not tgt_tx_los:
    #     print("No line of sight")
    #     return 0.0, 0.0

    ## Setting & Calculating of some constants which are independent of target position
    c = 299792458.0  # speed of light

    # dist_delay_limit [m], delay_thresh [us], baseline_range in [km] * 1000 in [m]
    dist_delay_limit = delay_thresh * c / 1e6 + (baseline_range * 1000)

    rcs, snr = calculate_min_rcs_without_los_single_pos(
        tgt.point,
        rx,
        tx,
        snr_thresh,
        dist_delay_limit,
        static_rcs,
        propagation_loss,
    )

    return rcs, snr


def calculate_minimum_detectable_rcs(
    Rx: Radar,
    Tx: Radar,
    point_of_interest: Point,
    dist_delay_limit: float,
    snr_threshold: float
):
    r"""
    Calculate the minimum radar cross section (RCS) that can be detected for the given geometry.

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
    The principle is described in the unpublished master thesis with the
    title "Passive Radar - From Quality Criteria to Coverage Optimization".
    Since the bistatic RCS :math:`\sigma_B` is unknown, the original expression
    for the SNR is reformulated and only the term :math:`SNR / \sigma_B` is used.

    .. math::

       RCS_{min} = 10^{(SNR_{B, threshold} - SNR / \sigma_B) / 10},
    
    where :math:`SNR_{B, threshold}` is the bistatic signal-to-noise ratio (SNR) threshold [dB],
    :math:`SNR` is the actual SNR [dB] and  is the unknown bistatic RCS.

    The SNR is calculated according to the radar equation

    .. math::

        \frac{SNR}{\sigma_B} = \frac{P_{T, EIRP} G_R G_p \lambda^2 F_T^2 F_R^2}{(4 \pi)^3 k_B T_S B_n L_T L_R} \frac{1}{R_T^2 R_R^2},

    where :math:`P{T, EIRP}` is the transmitter's EIRP,
    :math:`G_R, G_P` are the receiving antenna and processing gain,
    :math:`\lambda` is the signal wavelength, :math:`k_T T_S B_n` is the effective
    input noise power. The quantities :math:`F_T, F_R` are angle-dependent pattern
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

    attenuation = rx_horiz_att + tx_horiz_att + tx_vert_att + rx_vert_att

    # Calculate minimum detectable RCS.

    # TODO: Check formula!!!
    # Where are the distance terms???
    snr = (
        Tx.erp # Watt or db???
        + 2.15
        + Tx.processing_gain # dB or dBi???
        + Rx.gain # dB or dBi???
        - abs(Rx.losses) # dB or dBi???
        - 10 * math.log10(sc.Boltzmann * Rx.noise_temperature * Rx.bandwidth * 1000)
    )

    snr_extra = snr - snr_threshold - attenuation

    rcs = 10.0 ** (-snr_extra / 10.0)

    
    return np.clip(rcs, a_min=1e-6)



def calculate_min_rcs_without_los_single_pos(
    target_pos: Point,
    Rx: Radar,
    Tx: Radar,
    snr_thresh,
    dist_delay_limit,
    static_rcs,
    total_splat_loss,
):
    """
    Calculate min_rcs and snr for one single target position.

    Returns
    -------
    min_rcs: float
        ??? [m^2]
    snr: float
        Signal-to-noise ratio
    """
    r_r = (
        get_2d_distance_between_locs_heights(
            Rx.lat,
            Rx.lon,
            Rx.alt + Rx.antenna_height,
            target_pos.lat,
            target_pos.lon,
            target_pos.alt,
        )
        * 1000.0
    )  # distance in meters

    theta_r_bearing = get_azimuth_between_locs(
        Rx.lat,
        Rx.lon,
        target_pos.lat,
        target_pos.lon,
    )
    theta_r_vert = get_elev_angle(
        target_pos.alt,
        Rx.alt + Rx.antenna_height,
        r_r,
    )

    r_t = (
        get_2d_distance_between_locs_heights(
            Tx.lat,
            Tx.lon,
            Tx.alt + Tx.antenna_height,
            target_pos.lat,
            target_pos.lon,
            target_pos.alt,
        )
        * 1000.0
    )  # distance in meters

    theta_t_bearing = get_azimuth_between_locs(
        Tx.lat,
        Tx.lon,
        target_pos.lat,
        target_pos.lon,
    )
    theta_t_vert = get_elev_angle(
        target_pos.alt,
        Tx.alt + Tx.antenna_height,
        r_t,
    )

    if r_r + r_t < dist_delay_limit:  # checks if delay threshold is valid
        rcs = -1
        snr = -1
    else:
        tx_horiz_att = Tx.horizontal_attenuation(theta_t_bearing)
        rx_horiz_att = Rx.horizontal_attenuation(theta_r_bearing)
        tx_vert_att = Tx.vertical_attenuation(theta_t_vert)
        rx_vert_att = Rx.vertical_attenuation(theta_r_vert)

        # calculating the amount of snr which is above the threshold
        # the calculation of the snr is split into several parts to save computational time

        # propagation loss (includes besides propagation loss over terrain also freq dependent atmospheric attenutation) and a freq dependent free space loss computation by SPLAT are considered
        snr_extra = (
            snr_const_splat
            - snr_thresh
            - rx_horiz_att
            - tx_horiz_att
            - tx_vert_att
            - rx_vert_att
            - total_splat_loss
        )

        rcs = 10.0 ** (-snr_extra / 10.0)
        # see equation 2.20 in P. Mousel master thesis p.18
        if rcs == 0.0:  # avoid rcs being 0 and math domain error in log further down
            rcs = 0.000001
        # print("snr_extra, rcs: ", snr_extra, rcs)

        if static_rcs > 0.0:
            snr = snr_thresh + 10 * math.log10(static_rcs) - 10 * math.log10(rcs)
            # static_rcs is used just for snr computation with a static minimal rcs. we set static_rcs to 0 not to use this
        else:
            snr = snr_thresh - 10 * math.log10(rcs)

        if rcs > 150:  # we set the max to 150m2
            rcs = 150

    return rcs, snr
