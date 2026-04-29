import numba
import numpy as np
import scipy.constants as sc


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
    cpi_pulses: int,
    equivalent_temperature: float,
    L_t: float,
    L_r: float,
    L_a: float,
    polarization_factor: float,
    pattern_propagation_factor_transmitter: float,
    pattern_propagation_factor_receiver: float,
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
        Receiver bandwidth of the signal [MHz]
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
    pattern_propagation_factor_transmitter: float
        Pattern propagation factor for the path from the transmitter
        to the target [dB]
    pattern_propagation_factor_receiver: float
        Pattern propagation factor for the path from the target
        to the receiver [dB]
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
    lambda_sq_dB = 2 * 10 * np.log10(wavelength)
    rcs_dB = 10 * np.log10(radar_cross_section)
    four_pi_dB = 10 * np.log10(pow((4 * np.pi), 3 if not is_one_way else 2))
    ktb = 10 * np.log10(sc.Boltzmann * equivalent_temperature)

    bw_dB = 10 * np.log10(bandwidth * 1e6)

    snr = (
        power_dB
        + coherent_integration_gain_dB
        + antenna_gain_transmitter
        + antenna_gain_receiver
        + lambda_sq_dB
        + rcs_dB
        + 2 * polarization_factor
        + pattern_propagation_factor_transmitter
        + pattern_propagation_factor_receiver
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
    # print(f"pattern_propagation_factor_transmitter = {pattern_propagation_factor_transmitter:.3f} dB")
    # print(f"pattern_propagation_factor_receiver = {pattern_propagation_factor_receiver:.3f} dB")
    # print(f"SNR                       = {snr:.3f} dB")

    return snr
