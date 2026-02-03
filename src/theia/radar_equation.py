import math
import numpy as np
import scipy.constants as sc

from theia.types import Radar
from theia.util import marcum_q_function


# Taken from OpenBurst.
def radar_eq_max_dist(radar: Radar, target_rcs: float) -> float:
    """! returns maximal distance [m] given the radar parameters and the target rcs.
    MAX RANGE IS SET TO 400kms, due to the HARD LIMIT in SPLAT! (see MAXPAFES in splatBurst.h)
    """
    trans_pwr = radar.transmitter.power
    antenna_diam = radar.receiver.diameter
    frequency = radar.transmitter.frequency / 1000.
    pulse_width = radar.transmitter.pulse_width
    cpi_pulses = radar.receiver.cpi_pulses
    bandwidth = radar.transmitter.bandwidth
    pfa = radar.receiver.pfa
    rcsSM = target_rcs


    ####################### Radar and target values (working) ##################

    c = sc.speed_of_light
    # speed of light
    rf_loss = 12
    # RF system hardware loss (not known for ASR, best guess from chapter 2.12, p.80 Skolnik)
    rcs_start = 10 * np.log10(rcsSM)
    # rcs_start=13;                        # RCS in db
    noise_figure = 1.9
    # Receiver LNA noise figure in dB  (not known for TA, best guess)
    window = 0
    # rectangular for no window, or Hamming (=0)
    equiv_temp = 300
    # equivalent temperature [K] (not known for ASR, best guess)

    ########################################################################################

    ############# ------------- analyze for upto 400 kms
    start_range = 0.001
    # [m] starting at 0 will cause a division by 0 error further down
    # Attention!!!!!! if the snr is very high (that is for low ranges) the besseli function will overflow
    # throwing a segmentation fault (another way to avoid it is to return Pd=1 for snr > e.g. 30dB)
    # this can be avoided by setting the start_range to be a higher value
    stop_range = 400000
    # [m]

    # ------------------------------evaluate  range resolution ---------------

    res = c / (2 * (bandwidth * 1e6))
    if window == 0:
        res = res * 1.44

    # we set resolution manually
    res = 1000  # [m]

    # -----------------------evaluate radar range equation -----------------------------------

    A = np.pi * antenna_diam * antenna_diam / 4
    wavelength = c / (frequency * 1e9)
    antenna_gain = 10 * np.log10(0.6 * 4 * np.pi * A / (wavelength * wavelength))

    four_pi = 10 * np.log10(pow((4 * np.pi), 3))
    pt = 10 * np.log10(trans_pwr)
    lambda_sq = 2 * 10 * np.log10(c / (frequency * 1e9))
    ktb = 10 * np.log10(1.38e-23 * equiv_temp * (bandwidth * 1e6))
    t_bw_gain = 10 * np.log10(pulse_width * bandwidth)
    dop_gain = 10 * np.log10(cpi_pulses)

    ranges = np.arange(start_range, stop_range + 1, res)
    snr = []
    for rng in ranges:
        curr_snr = (
            pt
            + lambda_sq
            + 2 * antenna_gain
            + t_bw_gain
            + dop_gain
            + rcs_start
            - four_pi
            - ktb
            - 40 * np.log10(rng)
            - noise_figure
            - rf_loss
        )
        snr = np.concatenate([snr, [curr_snr]])

    # -----------------------------evaluate Pd function -----------------------

    beta = math.sqrt(-2 * (np.log(pfa)))

    rangel = np.arange(start_range, stop_range + 1, res)

    # lrl = rangel.shape[0]

    jj = 0
    pd = []

    for rng in rangel:
        # avoid segmentation fault in the besseli function for high snr values
        if snr[jj] > 30:
            pd = np.concatenate([pd, [1.0]])
        else:
            alpha = pow(10, (snr[jj] + 3) / 20)
            curr_pd = marcum_q_function(alpha, beta)
            pd = np.concatenate([pd, [curr_pd]])

        jj = jj + 1

    # compute and return max range (attention: all max ranges above this will not be noted by the user)
    retval = 400000

    ii = 0
    for _ in pd:
        if pd[ii] <= 0.8:
            retval = (rangel[ii] + rangel[ii - 1]) / 2
            break
        ii = ii + 1

    return retval
