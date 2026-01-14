# Utilitiy functions for passive radar detection.
import numpy as np
import scipy.constants as sc
from theia.coordinates import CoordinateTransformations
from theia.types import Radar, Target


def _calculate_additional_detection_infos(
    tx: Radar,
    rx: Radar,
    tgt: Target,
    snr: float,
    doppler: float,
    max_coherent_integration_time_fm: float,
) -> tuple[float, float, float]:
    # snr and snr_thresh in [dB]
    # (bistatic_dopppler_shift = (1/signal_wavelength) * bistatic_range_rate) (equation for bistatic doppler computation)
    # from the above you can derive: doppler velocity ()= bistatic_range_rate), where signal_wavelength = c/f
    bistatic_velocity = doppler * (
        sc.speed_of_light / (tx.frequency * 1_000_000)
    )  # [Hz] * [m/s]/[Hz] = [m/s]

    # calculate standard deviations of bistatic range and velocity
    rx_pos = CoordinateTransformations.geodetic_to_cartesian(
        rx.lat, rx.lon, rx.alt + rx.antenna_height
    )  # cartesian
    tx_pos = CoordinateTransformations.geodetic_to_cartesian(
        tx.lat, tx.lon, tx.alt + tx.antenna_height
    )  # cartesian
    tgt_pos = CoordinateTransformations.geodetic_to_cartesian(
        tgt.lat, tgt.lon, tgt.alt
    )  # cartesian
    snr_db = snr
    tx_freq_hz = tx.frequency * 1_000_000  # [MHz] to [Hz]
    tx_bw_hz = tx.bandwidth * 1_000_000  # [MHz] to [Hz]
    t_obs_s = max_coherent_integration_time_fm  # we use the FM case; TBD for other signal types

    # sigma_rho : standard deviation of bistatic range
    # sigma_v: standard deviation of bistatic radial velocity
    (sigma_rho, sigma_v) = calculate_std_devs_for_bistatic_detection(
        rx_pos, tx_pos, tgt_pos, snr_db, tx_freq_hz, tx_bw_hz, t_obs_s
    )
    sigma_rho = np.clip(
        sigma_rho, 10.0, 100.0
    )  # [m] clip range std dev to realistic values
    sigma_v = np.clip(sigma_v, 0.5, 5.0)  # [m/s] lip vel std dev to realistic values

    return bistatic_velocity, sigma_rho, sigma_v


def calculate_std_devs_for_bistatic_detection(
    rx_pos, tx_pos, tgt_pos, snr_db, tx_freq_hz, tx_bw_hz, t_obs_s
):
    """
    calculates range and vel std deviations of bistatic range and velocity calculations
    input rx_pos: rx position
    input tx_pos: tx position
    input tgt_pos: target position
    input snr_db: SNR of detection
    input tx_frq_hz: Tx freq [Hz]
    tx_bw_hz: signal bandwidth [Hz]
    t_obs_s: coherent integration time [s]

    returns:
    sigma_rho : standard deviation of bistatic range
    sigma_v: standard deviation of bistatic radial velocity
    """
    snr = 10 ** (snr_db / 10)
    beta = bistatic_angle(rx_pos, tx_pos, tgt_pos)
    beta_rms = 0.3 * tx_bw_hz

    sigma_tau = 1 / (2 * np.pi * beta_rms * np.sqrt(2 * snr))
    sigma_rho = sc.c * sigma_tau

    sigma_f = 1 / (2 * np.pi * t_obs_s * np.sqrt(2 * snr))
    lam = sc.c / tx_freq_hz
    sigma_v = (lam / (2 * np.cos(0.5 * beta))) * sigma_f

    return (sigma_rho, sigma_v)


def bistatic_angle(rx_pos, tx_pos, tgt_pos):
    """
    calculates bistatic_angle [rad] given rx, tx and tgt positions
    """
    rx_vec = np.array(rx_pos) - np.array(tgt_pos)
    tx_vec = np.array(tx_pos) - np.array(tgt_pos)
    cosb = np.dot(rx_vec, tx_vec) / (np.linalg.norm(rx_vec) * np.linalg.norm(tx_vec))
    return np.arccos(cosb)
