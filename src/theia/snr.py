import math

import numba
import numpy as np
import scipy.constants as sc

from theia.coordinates import calculate_azimuth_angle, calculate_elevation_angle
from theia.types import Point, Receiver, Transmitter


# Active radar and PCL.
@numba.njit
def calculate_snr(
    wavelength: float,
    antenna_gain_transmitter: float,
    antenna_gain_receiver: float,
    radar_cross_section: float,
    distance_transmitter_target: float,
    distance_receiver_target: float,
    transmission_power: float,
    bandwidth: float,
    pulse_width: float,
    cpi_pulses: int,
    equivalent_temperature: float,
    L_t: float,
    L_r: float,
    L_a: float,
    polarization_factor: float,
    antenna_pattern_gain_transmitter: float,
    antenna_pattern_gain_receiver: float,
    is_one_way: bool,
) -> float:
    r"""
    Calculate the signal-to-noise ratio (SNR) [dB] for free propagation.

    Parameters
    ----------
    wavelength: float
        Signal wavelength [m]
    antenna_gain_transmitter: float
        Antenna gain of the transmitter [dBi]
    antenna_gain_receiver: float
        Antenna gain of the receiver [dBi]
    radar_cross_section: float
        Radar cross section of the target [m^2]
    distance_transmitter_target: float
        Line-of-sight distance between the transmitter and the target [m]
    distance_receiver_target: float
        Line-of-sight distance between the receiver and the target [m]
    transmission_power: float
        Power of the signal [W]
    bandwidth: float
        Noise bandwidth of the signal [MHz]
    pulse_width: float
        Pulse width [us]
    cpi_pulses:
        Number of pulses within a Coherent Processing Interval (CPI)
    equivalent_temperature: float
        Equivalent temperature [K]
    L_t: float
        Transmission line loss [dB]
    L_r: float
        Receiver loss [dB]
    L_a: float
        Atmospheric and precipitation attenuation [dB]
    polarization_factor: float
        Polarization factor [dB]
    antenna_pattern_gain_transmitter: float
        Antenna pattern gain for the transmitter [dB]
    pattern_propagation_factor_receiver: float
        Antenna pattern gain for the receiver [dB]
    is_one_way: bool
        Whether the SNR calculation is for a signal travelling one-way
        (e. g. monostatic, PCL) or two-way (e. g. PET)

    Returns
    -------
    snr: float
        Signal-to-noise ratio

    Assumptions
    -----------
    - There is a direct line-of-sight.
    - No propagation losses.
    - No terrain losses.

    Notes
    -----
    Details about the implemented formula can be found in the docs at :ref:`snr-section`.
    The pulse width was replaced by the corresponding noise bandwidth
    :math:`B_n \approx \frac{1}{\tau}`.
    """
    # Scale to dB units.
    power_dB = 10 * np.log10(transmission_power)
    coherent_integration_gain_dB = 10 * np.log10(cpi_pulses)
    pulse_compression_gain = 10 * np.log10(pulse_width * bandwidth)
    lambda_sq_dB = 2 * 10 * np.log10(wavelength)
    rcs_dB = 10 * np.log10(radar_cross_section)
    four_pi_dB = 10 * np.log10(pow((4 * np.pi), 3 if not is_one_way else 2))
    ktb = 10 * np.log10(sc.Boltzmann * equivalent_temperature)

    bw_dB = 10 * np.log10(bandwidth * 1e6)

    snr = (
        power_dB
        + coherent_integration_gain_dB
        + pulse_compression_gain
        + antenna_gain_transmitter
        + antenna_gain_receiver
        + lambda_sq_dB
        + rcs_dB
        + 2 * polarization_factor
        + antenna_pattern_gain_transmitter
        + antenna_pattern_gain_receiver
        - four_pi_dB
        - ktb
        - bw_dB
        - 20 * np.log10(distance_transmitter_target)
        - 20 * np.log10(distance_receiver_target)
        - L_r
        - L_t
        - L_a
    )

    # print(f"power                     = {power_dB:.3f} dB")
    # print(f"coherent integration gain = {coherent_integration_gain_dB:.3f} dB")
    # print(f"Tx antenna gain           = {antenna_gain_transmitter:.3f} dB")
    # print(f"Rx antenna gain           = {antenna_gain_receiver:.3f} dB")
    # print(f"lambda                    = {wavelength:.3f} m")
    # print(f"lambda^2                  = {lambda_sq_dB:.3f} dB")
    # print(f"RCS                       = {rcs_dB:.3f} dB")
    # print(f"(4pi)^3                   = {four_pi_dB:.3f} dB")
    # print(f"k_B T                     = {ktb:.3f} dB")
    # print(f"Bandwidth                 = {bw_dB:.3f} dB")
    # print(f"Polarization factor       = {polarization_factor:.13f} dB")
    # print(f"r_t                       = {distance_transmitter_target:.3f} m")
    # print(f"r_r                       = {distance_receiver_target:.3f} m")
    # print(f"L_t                       = {L_t:.3f} dB")
    # print(f"L_a                       = {L_a:.3f} dB")
    # print(f"antenna_pattern_gain_transmitter = {antenna_pattern_gain_transmitter:.3f} dB")
    # print(f"antenna_pattern_gain_receiver = {antenna_pattern_gain_receiver:.3f} dB")
    # print(f"SNR                       = {snr:.3f} dB")

    return snr


def calculate_antenna_pattern_gain(
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
    theta_bearing = (theta_bearing + 2 * math.pi) % (2 * math.pi)
    theta_vert = (theta_vert + 2 * math.pi) % (2 * math.pi)

    horiz_att = (
        transmitter_or_receiver.horizontal_attenuation(theta_bearing)
        if transmitter_or_receiver.horizontal_attenuation is not None
        else 0.0
    )
    vert_att = (
        transmitter_or_receiver.vertical_attenuation(theta_vert)
        if transmitter_or_receiver.vertical_attenuation is not None
        else 0.0
    )

    return -horiz_att - vert_att
