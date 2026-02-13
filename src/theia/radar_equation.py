import scipy.constants as sc

from theia.config import RF_LOSS
from theia.detection.active import calculate_probability_of_detection
from theia.snr import calculate_snr
from theia.types import Radar
from theia.util import get_clear_sky_attenuation


def calculate_maximum_monostatic_range(
    radar: Radar,
    target_rcs: float,
    probability_threshold: float = 0.8,
    maximum_expected_range: float = 2**20,
    resolution: float = 1.0,
    rf_loss: float = RF_LOSS,
) -> float:
    r"""
    Maximum range [m] such that a minimum probability of detection is achieved.

    Parameters
    ----------
    radar: Radar
        Radar
    target_rcs: float
        RCS of a target to be detected [m^2]
    probability_threshold: float, default 0.8
        Minimum detection probability in [0, 1] to be achieved at the maximum range
    maximum_expected_range: float, default 2**20 :math:`\approx.` 1000km
        Maximum range to be assumed. Determines the number of queries to be made.
    resolution: float, default 1.0
    rf_loss: float, default RF_LOSS
        Loss of the transmitter (correct? TODO)

    Returns
    -------
    float
        Maximum range at which a target of the given RCS can be detected

    See also
    --------
    Details about the implemented formula can be found in the docs at :ref:`snr-section`.
    The pulse width was replaced by the corresponding noise bandwidth
    :math:`B_n \approx \frac{1}{\tau}`.

    Notes
    -----
    Algorithm: We start at range zero. A bisection approach is taken.
    As long as the range is still detectable, we take one step further.
    In every iteration, the step width is halved.
    """
    wavelength = sc.speed_of_light / (radar.transmitter.frequency * 1e6)

    # Factor 2 is needed because the signal travels both ways.
    atmospheric_loss_per_distance = (
        get_clear_sky_attenuation(radar.transmitter.frequency) * 2 / 1000.0
    )

    def calc_pd(r: float) -> float:
        snr = calculate_snr(
            wavelength=wavelength,
            antenna_gain_transmitter=radar.transmitter.antenna_gain,
            antenna_gain_receiver=radar.receiver.antenna_gain(
                radar.transmitter.frequency
            ),
            radar_cross_section=target_rcs,
            distance_transmitter_target=r,
            distance_receiver_target=r,
            transmission_power=radar.transmitter.power,
            bandwidth=radar.receiver.bandwidth,
            cpi_pulses=radar.receiver.cpi_pulses,
            equivalent_temperature=radar.receiver.noise_temperature,
            L_t=rf_loss,
            L_a=atmospheric_loss_per_distance * r,
            # TODO: Should we include these factors?
            polarization_factor=0.0,
            pattern_propagation_factor_receiver=0.0,
            pattern_propagation_factor_transmitter=0.0,
        )
        # Avoid blowup in Bessel function.
        if snr > 30:
            p = 1.0
        else:
            p = calculate_probability_of_detection(snr, radar.receiver.pfa)
        return p

    step = 0.5 * maximum_expected_range
    r = 0
    while step >= 0.5 * resolution:
        r_further = r + step
        p_further = calc_pd(r_further)
        if p_further >= probability_threshold:
            r += step
        step /= 2

    return r
