import numba
import numpy as np
import scipy.constants as sc


def calculate_thermal_noise_loss(
    equivalent_temperature: float, bandwidth: float
) -> float:
    """
    Calculate the thermal noise loss [dB].

    Parameters
    ----------
    equivalent_temperature: float
        Noise temperature [K].
    bandwidth: float
        Signal bandwidth [MHz]
    """
    return 10 * np.log10(sc.Boltzmann * equivalent_temperature * (bandwidth * 1e6))


# @numba.njit
# def calculate_snr_simple(
#     frequency: float,
#     radar_cross_section: float,
#     distance_transmitter_target: float,
#     distance_receiver_target: float,
#     transmission_power: float,
#     gain_transmitter: float,
#     gain_receiver: float,
#     loss_transmitter: float,
#     loss_receiver: float,
# ) -> float:
#     r"""
#     Calculate the signal-to-noise ratio (SNR) for given radar properties for free propagation.

#     Parameters
#     ----------
#     frequency: float
#         Signal frequency [GHz]
#     radar_cross_section: float
#         Radar cross section of the target [m^2]
#     distance_transmitter_target: float
#         Line-of-sight distance between the target and the transmitter [km]
#     distance_receiver_target: float
#         Line-of-sight distance between the target and the receiver [km]
#     transmission_power: float
#         Power of the signal [W]
#     gain_transmitter: float
#         Total gains of the transmitter [dB]
#     gain_receiver: float
#         Total gains of the receiver [dB]
#     loss_transmitter: float
#         Total losses of the transmitter [dB]
#     loss_receiver: float
#         Total losses of the receiver [dB]

#     Returns
#     -------
#     snr: float
#         Signal-to-noise ratio

#     Assumptions
#     -----------
#     - There is a direct line-of-sight.
#     - No propagation losses.
#     - No terrain losses.

#     Notes
#     -----
#     The following formula is implemented:

#     .. math::
#        SNR = \frac{P \lambda^2 G \sigma}{(4 \pi)^3 k_B T B R_T^2 R_R^2 L},

#     where :math:`P` is the transmitter power [W], :math:`\lambda` is the signal
#     wave length, :math:`G_{T}, G_{R}` are the transmitter and
#     receiver gains, :math:`G_{pulse compression}` is the pulse compression gain,
#     :math:`G_{coherent integration}` is the gain from coherent integration,
#     :math:`\sigma` is the radar cross section of the target, :math:`T` is the
#     noise temperature, :math:`B` is the bandwidth, :math:`R_T, R_R` are the distance
#     between transmitter / receiver and the target, :math:`F` is the noise figure
#     and :math:`L` is a collective term for all other kinds of RF losses.

#     The antenna gain is assumed to be the same for the transmitter and receiver
#     and estimated using the formula

#     .. math::

#        G_{T} = G_{R} = 0.6 \cdot \pi (\frac{D}{2})^2 \frac{4 \pi}{\lambda^2}.
#     """
#     wavelength = sc.speed_of_light / (frequency * 1e9)

#     # Scale to dB units.
#     rcs_dB = 10 * np.log10(radar_cross_section)
#     four_pi_dB = 10 * np.log10(pow((4 * np.pi), 3))
#     power_dB = 10 * np.log10(transmission_power)
#     lambda_sq_dB = 2 * 10 * np.log10(wavelength)

#     return (
#         power_dB
#         + lambda_sq_dB
#         + rcs_dB
#         - four_pi_dB
#         - 20 * np.log10(distance_transmitter_target)
#         - 20 * np.log10(distance_receiver_target)
#         + gain_transmitter
#         + gain_receiver
#         - loss_transmitter
#         - loss_receiver
#     )


# @numba.njit
# def calculate_snr(
#     frequency: float,
#     radar_cross_section: float,
#     distance_transmitter_target: float,
#     distance_receiver_target: float,
#     transmission_power: float,
#     bandwidth: float,
#     equivalent_temperature: float,
#     noise_figure: float,
#     rf_loss: float,
#     G_pulse_compression: float,
#     G_coherent_integration,
#     G_r: float,
#     G_t: float,
# ) -> float:
#     r"""
#     Calculate the signal-to-noise ratio (SNR) for given radar properties for free propagation.

#     Parameters
#     ----------
#     frequency: float
#         Signal frequency [GHz]
#     radar_cross_section: float
#         Radar cross section of the target [m^2]
#     distance_transmitter_target: float
#         Line-of-sight distance between the target and the transmitter [km]
#     distance_receiver_target: float
#         Line-of-sight distance between the target and the receiver [km]
#     transmission_power: float
#         Power of the signal [W]
#     bandwidth: float
#         Bandwidth of the signal [MHz]
#     antenna_diameter: float
#         Antenna diameter [m^2]
#     equivalent_temperature: float
#         Equivalent temperature [K]
#     noise_figure: float
#         Receiver LNA noise figure [dB]
#     rf_loss: float
#         RF system hardware loss [dB]
#     G_pulse_compression: float
#         Pulse compression gain [dB]
#     G_coherent_integration: float
#         Coherent integration gain [dB]
#     G_r: float
#         Antenna gain of the receiver [db]
#     G_t: float
#         Antenna gain of the transmitter [db]

#     Returns
#     -------
#     snr: float
#         Signal-to-noise ratio

#     Assumptions
#     -----------
#     - There is a direct line-of-sight.
#     - No propagation losses.
#     - No terrain losses.

#     Notes
#     -----
#     The following formula is implemented:

#     .. math::
#        SNR = \frac{P \lambda^2 G_{T} G_{R} G_{pulse compression} G_{coherent integration} \sigma}{(4 \pi)^3 k_B T B R_T^2 R_R^2 F L},

#     where :math:`P` is the transmitter power [W], :math:`\lambda` is the signal
#     wave length, :math:`G_{T}, G_{R}` are the transmitter and
#     receiver gains, :math:`G_{pulse compression}` is the pulse compression gain,
#     :math:`G_{coherent integration}` is the gain from coherent integration,
#     :math:`\sigma` is the radar cross section of the target, :math:`T` is the
#     noise temperature, :math:`B` is the bandwidth, :math:`R_T, R_R` are the distance
#     between transmitter / receiver and the target, :math:`F` is the noise figure
#     and :math:`L` is a collective term for all other kinds of RF losses.

#     The antenna gain is assumed to be the same for the transmitter and receiver
#     and estimated using the formula

#     .. math::

#        G_{T} = G_{R} = 0.6 \cdot \pi (\frac{D}{2})^2 \frac{4 \pi}{\lambda^2}.
#     """
#     wavelength = sc.speed_of_light / (frequency * 1e9)

#     # Scale to dB units.
#     rcs_dB = 10 * np.log10(radar_cross_section)
#     four_pi_dB = 10 * np.log10(pow((4 * np.pi), 3))
#     power_dB = 10 * np.log10(transmission_power)
#     lambda_sq_dB = 2 * 10 * np.log10(wavelength)
#     thermal_noise_loss_dB = 10 * np.log10(
#         sc.Boltzmann * equivalent_temperature * (bandwidth * 1e6)
#     )

#     return (
#         power_dB
#         + lambda_sq_dB
#         + G_t
#         + G_r
#         + G_pulse_compression
#         + G_coherent_integration
#         + rcs_dB
#         - four_pi_dB
#         - thermal_noise_loss_dB
#         - 20 * np.log10(distance_transmitter_target)
#         - 20 * np.log10(distance_receiver_target)
#         - noise_figure
#         - rf_loss
#     )


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
    Calculate the signal-to-noise ratio (SNR) for given radar properties for free propagation.

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
