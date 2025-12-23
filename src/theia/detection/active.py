import math
import numba
import numpy as np
import scipy.constants as sc
from scipy import integrate
from theia.distance import line_of_sight_distance
from theia.doppler import monostatic_doppler
from theia.line_of_sight import has_line_of_sight
from theia.radar_equation import marcum_q_fn
from theia.types import Radar, Target


def get_rad_pd(
    radar: Radar,
    target: Target,
    distance_step: float = 30.0,
    doppler_shift_threshold_hz: float = 5.0,
) -> float:
    rad_lat = radar.lat
    rad_lon = radar.lon
    rad_height = radar.alt
    power = radar.power
    antenna_diam = radar.diameter
    # The rest of the code except Doppler assumes GHz. (taken from openBURST)
    freq = radar.frequency / 1000.0
    pulse_width = radar.pulse_width
    cpi_pulses = radar.cpi_pulses
    bandwidth = radar.bandwidth
    pfa = radar.pfa

    tgt_lat = target.lat
    tgt_lon = target.lon
    tgt_height = target.alt
    rcs = target.cross_section

    # The rest of the code assumes dist to be in km. (taken from openBURST)
    dist = (
        line_of_sight_distance(
            rad_lat, rad_lon, rad_height, tgt_lat, tgt_lon, tgt_height
        )
        / 1000.0
    )

    # TODO: Adjust to use splat!
    los_ok = has_line_of_sight(radar.point, target.point, distance_step)

    if not los_ok:
        return 0.0
    else:
        doppler = monostatic_doppler(
            radar.frequency,  # yes, it is MHz here!
            rad_lat,
            rad_lon,
            radar.alt,
            tgt_lat,
            tgt_lon,
            target.alt,
            target.vlon,
            target.vlat,
            target.vz,
        )
        if doppler <= doppler_shift_threshold_hz:
            return 0.0

    got_pd = radar_detection_given_with_splat(
        power,
        antenna_diam,
        freq,
        pulse_width,
        cpi_pulses,
        bandwidth,
        pfa,
        rcs,
        dist,
    )

    return got_pd


@numba.njit
def calculate_snr(
    frequency: float,
    radar_cross_section: float,
    tgt_rad_dist: float,
    transmission_power: float,
    bandwidth: float,
    pulse_width: float,
    antenna_diameter: float,
    cpi_pulses: int,
    equivalent_temperature: float,
    noise_figure: float,
    rf_loss: float,
) -> float:
    """
    Calculate the signal-to-noise ratio (SNR) for given radar properties for free propagation.

    Parameters
    ----------
    frequency: float
        Signal frequency in GHz
    radar_cross_section: float
        Radar cross section of the target in m^2
    tgt_rad_dist: float
        Line-of-sight distance between the target and the transmitter in km
    transmission_power: float
        Power of the signal in W
    bandwidth: float
        Bandwidth of the signal in m
    pulse_width: float
        Pulse width in us (micro seconds)
    antenna_diameter: float
        Antenna diameter in m^2
    cpi_pulses:
        Number of pulses within a Coherent Processing Interval (CPI)
    equivalent_temperature: float
        Equivalent temperature in K
    noise_figure: float
        Receiver LNA noise figure in dB
    rf_loss: float
        RF system hardware loss in dB

    Returns
    -------
    snr: float
        Signal-to-noise ratio

    Assumptions
    -----------
    - There is a direct line-of-sight.
    - No propagation losses.
    - No terrain losses.
    """
    wavelength = sc.speed_of_light / (frequency * 1e9)

    # Scale to dB units.
    rcs_start = 10 * np.log10(radar_cross_section)
    four_pi = 10 * np.log10(pow((4 * np.pi), 3))
    pt = 10 * np.log10(transmission_power)
    lambda_sq = 2 * 10 * np.log10(wavelength)
    ktb = 10 * np.log10(1.38e-23 * equivalent_temperature * (bandwidth * 1e6))

    ########################################################################################

    ############# ------------- analyze for upto 400 kms
    # start_range = 0
    # [m]
    # Attention!!!!!! if the snr is very high (that is for low ranges) the besseli function will overflow
    # throwing a segmentation fault (another way to avoid it is to return Pd=1 for snr > e.g. 30dB)
    # this can be avoided by setting the start_range to be a higher value
    # stop_range = 400000
    # [m]

    # -----------------------evaluate radar range equation -----------------------------------

    antenna_area = np.pi * antenna_diameter * antenna_diameter / 4
    antenna_gain = 10 * np.log10(
        0.6 * 4 * np.pi * antenna_area / (wavelength * wavelength)
    )

    pt = 10 * np.log10(transmission_power)
    lambda_sq = 2 * 10 * np.log10(wavelength)

    t_bw_gain = 10 * np.log10(pulse_width * bandwidth)
    dop_gain = 10 * np.log10(cpi_pulses)
    range_of_target = tgt_rad_dist

    return (
        pt
        + lambda_sq
        + 2 * antenna_gain
        + t_bw_gain
        + dop_gain
        + rcs_start
        - four_pi
        - ktb
        - 40 * np.log10(range_of_target)
        - noise_figure
        - rf_loss
    )


def radar_detection_given_with_splat(
    trans_pwr: float,  # [W]
    antenna_diam: float,  # [m]
    frequency: float,  # [GHz]
    pulse_width: float,  # [us]
    cpi_pulses: int,  # [no units]
    bandwidth: int,  # [MHz]
    pfa: float,  # probability of false alarm [no units]
    rcs_sm: float,  # [m^2]
    tgt_rad_dist: float,  # [km]
    window: bool = False,  # rectangular for no window, or Hamming (=0)
    rf_loss: float = 12.0,  # RF system hardware loss [dB] (not known for ASR, best guess from chapter 2.12, p.80 Skolnik "Introduction to radar systems")
    noise_figure: float = 1.9,  # Receiver LNA noise figure [dB]  (not known for specific radars, best guess)
    equiv_temp: float = 300.0,  # Equivalent temperature [K]
) -> float:
    """
    returns pd given the radar parameters and the target rcs and the distance [m] between radar and target.
    if one_way_pure_prop_loss_db is given: we do not compute prop loss in both directions
    (Rad->Tgt and Tgt->Rad, instead we assume the prop loss is the same in both directions)

    This uses the results from splat computation to determine the radar detection:
    used for LIVE detection and by coverage computation with prop model.
    """
    snr = calculate_snr(
        frequency,
        rcs_sm,
        tgt_rad_dist,
        trans_pwr,
        bandwidth,
        pulse_width,
        antenna_diam,
        cpi_pulses,
        equiv_temp,
        noise_figure,
        rf_loss,
    )

    # ------------------------------evaluate  range resolution ---------------

    res = sc.speed_of_light / (2 * (bandwidth * 1e6))
    if not window:
        res = res * 1.44
        # -----------------------------evaluate Pd function -----------------------

        beta = math.sqrt(-2 * (np.log(pfa)))

        # avoid segmentation fault in the besseli function for high snr values
        if snr > 30:
            pd = 1.0
        else:
            alpha = pow(10, ((snr + 3) / 20))
            pd = (
                1
                - integrate.quad(
                    lambda x: marcum_q_fn(x, alpha),
                    0,
                    beta,
                )[0]
            )
        return pd
    else:
        raise NotImplementedError()
