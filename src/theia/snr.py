import numba
import numpy as np
import scipy.constants as sc


# Active radar.
@numba.njit
def calculate_snr(
    wavelength: float,
    antenna_gain_transmitter: float,
    antenna_gain_receiver: float,
    radar_cross_section: float,
    tgt_rad_dist: float,
    transmission_power: float,
    bandwidth: float,
    cpi_pulses: int,
    equivalent_temperature: float,
    noise_figure: float,
    rf_loss: float,
) -> float:
    r"""
    Calculate the signal-to-noise ratio (SNR) [dB] for free propagation.

    Parameters
    ----------
    wavelength: float
        Signal wavelength [m]
    antenna_gain_transmitter: float
        Antenna gain of the transmitter [dB]
    antenna_gain_receiver: float
        Antenna gain of the receiver [dB]
    radar_cross_section: float
        Radar cross section of the target [m^2]
    tgt_rad_dist: float
        Line-of-sight distance between the target and the transmitter [m]
    transmission_power: float
        Power of the signal [W]
    bandwidth: float
        Bandwidth of the signal [MHz]
    cpi_pulses:
        Number of pulses within a Coherent Processing Interval (CPI)
    equivalent_temperature: float
        Equivalent temperature [K]
    noise_figure: float
        Receiver LNA noise figure [dB]
    rf_loss: float
        RF system hardware loss [dB]

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
    The following formula is implemented (Skolnik 1980, Equ. 2.54):

    .. math::
       SNR &= \frac{P G_{T} A \rho \sigma n E_i(n)}{(4 \pi)^2 k_B T B R_T^2 R_R^2 F L}\\
           &= \frac{P \lambda^2 G_{T} G_{R} G_{coherent integration} \sigma}{(4 \pi)^3 k_B T B R_T^2 R_R^2 F L},

    where :math:`P` is the transmitter power [W], :math:`\lambda` is the signal
    wave length, :math:`G_{T}, G_{R}` are the transmitter and
    receiver gains, :math:`G_{coherent integration} = n E_i(n)` is the gain
    from coherent integration (:math:`n` denotes the number of integrated hits
    and :math:`E_i(n)` the integration efficiency),
    :math:`\sigma` is the radar cross section of the target, :math:`T` is the
    noise temperature, :math:`B` is the bandwidth, :math:`R_T, R_R` are the distance
    between transmitter / receiver and the target, :math:`F` is the noise figure
    and :math:`L` is a collective term for all other kinds of RF losses.

    References
    ----------
    Skolnik, M. I. (1980). Introduction to Radar Systems (2nd ed.). McGraw-Hill.
    """
    # Scale to dB units.
    power_dB = 10 * np.log10(transmission_power)
    lambda_sq_dB = 2 * 10 * np.log10(wavelength)
    rcs_dB = 10 * np.log10(radar_cross_section)
    four_pi_dB = 10 * np.log10(pow((4 * np.pi), 3))
    ktb = 10 * np.log10(sc.Boltzmann * equivalent_temperature)

    bw_dB = 10 * np.log10(bandwidth * 1e6)
    coherent_integration_gain_dB = 10 * np.log10(cpi_pulses)
    range_of_target = tgt_rad_dist

    return (
        power_dB
        + lambda_sq_dB
        + antenna_gain_transmitter
        + antenna_gain_receiver
        + coherent_integration_gain_dB
        + rcs_dB
        - four_pi_dB
        - ktb
        - bw_dB
        - 40 * np.log10(range_of_target)
        - noise_figure
        - rf_loss
    )
