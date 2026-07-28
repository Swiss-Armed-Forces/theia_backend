import functools

import scipy.constants as sc

from theia.config import RCS_FOR_RANGE_CALCULATION, RF_LOSS
from theia.detection.active import calculate_probability_of_detection
from theia.snr import calculate_snr
from theia.types import MonostaticSensor, PetSensor
from theia.util import get_clear_sky_attenuation


def calculate_maximum_monostatic_range(
    radar: MonostaticSensor | PetSensor,
    target_rcs: float = RCS_FOR_RANGE_CALCULATION,
    probability_threshold: float = 0.8,
    maximum_expected_range: float = 2**20,
    resolution: float = 1.0,
    rf_loss: float = RF_LOSS,
) -> float:
    r"""
    Maximum range [m] such that a minimum probability of detection is achieved.

    Parameters
    ----------
    radar: MonostaticSensor | PetSensor
        Monostatic radar or PET sensor
    target_rcs: float, default RCS_FOR_RANGE_CALCULATION
        RCS of a target to be detected [m^2]
    probability_threshold: float, default 0.8
        Minimum detection probability in [0, 1] to be achieved at the maximum range
    maximum_expected_range: float, default 2**20 :math:`\approx.` 1000km
        Maximum range to be assumed. Determines the number of queries to be made.
    resolution: float, default 1.0
        range resolution [m]
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
    return _calculate_maximum_monostatic_range(
        radar.transmitter.frequency,
        radar.transmitter.antenna_gain,
        radar.receiver.antenna_gain(radar.transmitter.frequency),
        target_rcs,
        radar.transmitter.power,
        radar.receiver.bandwidth,
        radar.transmitter.pulse_width,
        radar.receiver.cpi_pulses,
        radar.receiver.noise_temperature,
        rf_loss,
        radar.receiver.noise_figure,
        radar.receiver.pfa,
        maximum_expected_range,
        resolution,
        probability_threshold,
        type(radar) is PetSensor,
    )


@functools.lru_cache
def _calculate_maximum_monostatic_range(
    frequency: float,
    transmitter_antenna_gain: float,
    receiver_antenna_gain: float,
    target_rcs: float,
    power: float,
    bandwidth: float,
    pulse_width: float,
    cpi_pulses: int,
    noise_temperature: float,
    rf_loss: float,
    receiver_noise_figure: float,
    pfa: float,
    maximum_expected_range: float,
    resolution: float,
    probability_threshold: float,
    is_one_way: bool,
) -> float:
    wavelength = sc.speed_of_light / (frequency * 1e6)

    # Factor 2 is needed because the signal travels both ways.
    atmospheric_loss_per_distance = get_clear_sky_attenuation(frequency) * 2 / 1000.0

    def calc_pd(r: float) -> float:
        snr = calculate_snr(
            wavelength=wavelength,
            antenna_gain_transmitter=transmitter_antenna_gain,
            antenna_gain_receiver=receiver_antenna_gain,
            radar_cross_section=target_rcs,
            distance_transmitter_target=r,
            distance_receiver_target=r,
            transmission_power=power,
            bandwidth=bandwidth,
            pulse_width=pulse_width,
            cpi_pulses=cpi_pulses,
            equivalent_temperature=noise_temperature,
            L_t=rf_loss,
            L_r=receiver_noise_figure,
            L_a=atmospheric_loss_per_distance * r,
            # TODO: Should we include these factors?
            polarization_factor=0.0,
            pattern_propagation_factor_receiver=0.0,
            pattern_propagation_factor_transmitter=0.0,
            is_one_way=is_one_way,
        )
        p = calculate_probability_of_detection(snr, pfa)
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
