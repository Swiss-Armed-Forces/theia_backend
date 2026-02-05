import datetime

import numpy as np
import scipy.constants as sc

from theia.config import (
    DELAY_THRESHOLD_PCL,
    DOPPLER_SHIFT_THRESHOLD_PCL,
    SNR_THRESHOLD_PCL,
)
from theia.coordinates import calculate_azimuth_angle
from theia.distance import (
    get_2d_distance_between_locs_heights,
    get_bistatic_range,
    get_elev_angle,
)
from theia.doppler import calculate_bistatic_doppler
from theia.types import (
    PassiveRadarDetection,
    Point,
    RcsModel,
    Receiver,
    Target,
    Transmitter,
)
from theia.util import to_dB


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
    doppler = calculate_bistatic_doppler(rx, tgt, tx)  # [Hz]

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
        erp=tx.erp,
        wavelength=sc.speed_of_light / (tx.frequency * 1e6),
        distance_receiver_target=r_r,
        distance_transmitter_target=r_t,
        antenna_pattern_loss=calculate_antenna_pattern(
            rx,
            tx,
            tgt.point,
            r_t,
            r_r,
        ),
        noise_temperature=rx.noise_temperature,
        antenna_gain_receiver=rx.gain,
        transmitter_processing_gain=tx.processing_gain,
        receiver_losses=rx.losses,
        bandwidth=rx.bandwidth,
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


def calculate_antenna_pattern(
    Rx: Receiver,
    Tx: Transmitter,
    point_of_interest: Point,
    distance_transmitter_target: float,
    distance_receiver_target: float,
) -> float:
    """
    Calculate antenna pattern attenuation [dB] for the given geometry.
    """
    theta_t_bearing = calculate_azimuth_angle(
        p_observer=Tx.point,
        p_target=point_of_interest,
    )
    theta_t_vert = get_elev_angle(
        point_of_interest.alt,
        Tx.alt + Tx.antenna_height,
        distance_transmitter_target,
    )
    theta_r_bearing = calculate_azimuth_angle(
        p_observer=Rx.point,
        p_target=point_of_interest,
    )
    theta_r_vert = get_elev_angle(
        point_of_interest.alt,
        Rx.alt + Rx.antenna_height,
        distance_receiver_target,
    )

    tx_horiz_att = Tx.horizontal_attenuation(theta_t_bearing)
    rx_horiz_att = Rx.horizontal_attenuation(theta_r_bearing)
    tx_vert_att = Tx.vertical_attenuation(theta_t_vert)
    rx_vert_att = Rx.vertical_attenuation(theta_r_vert)

    return tx_horiz_att + rx_horiz_att + tx_vert_att + rx_vert_att


def calculate_snr(
    erp: float,
    wavelength: float,
    distance_receiver_target: float,
    distance_transmitter_target: float,
    antenna_pattern_loss: float,
    noise_temperature: float,
    antenna_gain_receiver: float,
    transmitter_processing_gain: float,
    receiver_losses: float,
    bandwidth: float,
):
    r"""
    Calculate the signal-to-noise ratio (SNR) [dB].

    Parameters
    ----------
    erp: float
        Effective Radiated Power [W]
    wavelength: float
        Wavelength [m]
    distance_receiver_target: float
        Distance between the receiver and the target [m]
    distance_transmitter_target: float
        Distance between the transmitter and the target [m]
    antenna_pattern_loss: float
        Loss due to the antenna pattern [dB]
    noise_temperature: float
        Noise temperature of the receiver [K]
    antenna_gain_receiver: float
        Antenna of the receiver [dBi]
    transmitter_processing_gain: float
        Processing gain of the transmitter [dBi]
    bandwidth: float
        Noise bandwidth of the receiver [MHz]
    receiver_losses: float
        Other losses of the receiver [dB]

    Notes
    -----
    The SNR is calculated according to the bistatic radar equation (Skolnik 1980, Equ. 14.36), extended by a thermal noise term (Skolnik 1980, Equ. 2.2) and a generic antenna pattern loss term,

    .. math::

        SNR_{\setminus \sigma_B} := \frac{SNR}{\sigma_B} = \frac{P_{T, EIRP} G_R G_p \lambda^2 F_T^2 F_R^2}{(4 \pi)^3 k_B T_S B_n L_T L_R} \frac{1}{R_T^2 R_R^2},

    where :math:`P{T, EIRP}` is the transmitter's EIRP,
    :math:`G_R, G_P` are the receiving antenna and processing gain,
    :math:`\lambda` is the signal wavelength, :math:`k_T T_S B_n` is the effective
    input noise power. The quantities :math:`F_T, F_R` are angle- and frequency-dependent pattern
    propagation factors.

    The quantities :math:`R_T, R_R` represent the distance
    between transmitter and point of interest as well as receiver and point of interest.

    Finally, the quantities :math:`L_T, L_R` represent other losses,
    including atmospheric absorption and line-feed losses between transmitter output
    and transmitter antenna as well as between receiving antenna output to receiver input.

    References
    ----------
    Skolnik, M. I. (1980). Introduction to Radar Systems (2nd ed.). McGraw-Hill.
    """
    # Evaluate attenuation for the angles of gaze.

    # Calculate SNR components in dB.
    eirp_dBW = to_dB(erp) + 2.15
    rx_thermal_noise_loss_dB = 10 * np.log10(
        (4 * np.pi) ** 3 * sc.Boltzmann * noise_temperature * bandwidth * 1e6
    )
    # frequency = sc.speed_of_light / wavelength
    # atmospheric_loss_dB = get_clear_sky_attenuation(frequency / 1e6) * (
    #     (distance_receiver_target + distance_transmitter_target) / 1000.0
    # )
    free_space_loss_dB = 20 * np.log10(
        distance_receiver_target * distance_transmitter_target
    )
    wavelength_squared_dB = 20 * np.log10(wavelength)

    return (
        eirp_dBW
        + antenna_gain_receiver
        + transmitter_processing_gain
        + wavelength_squared_dB
        - receiver_losses
        - rx_thermal_noise_loss_dB
        - antenna_pattern_loss
        # - atmospheric_loss_dB
        - free_space_loss_dB
    )
