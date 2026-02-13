import datetime
import numpy as np
import scipy.constants as sc

from theia.config import RF_LOSS
from theia.coordinates import calculate_azimuth_angle, calculate_elevation_angle
from theia.detection.active import calculate_probability_of_detection
from theia.distance import line_of_sight_distance
from theia.line_of_sight import has_line_of_sight
from theia.snr import calculate_pet_snr
from theia.types import PetDetection, Radar, Target
from theia.util import get_clear_sky_attenuation


def calculate_pet_detection(
    radar: Radar,
    target: Target,
    rng: np.random.Generator,
    distance_step: float = 30,
    rf_loss: float = RF_LOSS,
) -> PetDetection | None:

    los_ok = has_line_of_sight(
        radar.transmitter.point,
        radar.receiver.point,
        distance_step,
    )

    if not los_ok:
        return

    dist = line_of_sight_distance(
        *radar.receiver.point.as_tuple(),
        *radar.transmitter.point.as_tuple(),
    )

    wavelength = sc.speed_of_light / (radar.transmitter.frequency * 1e6)

    # We can "deactivate" some terms by setting them to one because the SNR
    # formula is multiplicative.
    snr_dB = calculate_pet_snr(
        wavelength=wavelength,
        antenna_gain_transmitter=radar.transmitter.antenna_gain,
        antenna_gain_receiver=radar.receiver.antenna_gain(radar.transmitter.frequency),
        distance_receiver_target=dist,
        transmission_power=radar.transmitter.power,
        bandwidth=radar.transmitter.bandwidth,
        cpi_pulses=1,
        equivalent_temperature=radar.receiver.noise_temperature,
        L_t=rf_loss,
        L_a=get_clear_sky_attenuation(radar.transmitter.frequency) * 2 * dist / 1000.0,
        # TODO: Should we include these factors?
        polarization_factor=0.0,
        pattern_propagation_factor_receiver=0.0,
        pattern_propagation_factor_transmitter=0.0,
    )

    p = calculate_probability_of_detection(snr_dB, radar.receiver.pfa)

    if rng.uniform(low=0, high=1) <= p:
        return PetDetection(
            detection_id=-1,
            time=datetime.datetime.fromtimestamp(0),
            radar=radar,
            target=target,
            target_range=line_of_sight_distance(
                *radar.transmitter.point.as_tuple(),
                *target.point.as_tuple(),
            ),
            elevation_angle=calculate_elevation_angle(
                radar.transmitter.point,
                target.point,
            ),
            azimuth_angle=calculate_azimuth_angle(
                radar.transmitter.point,
                target.point,
            ),
        )
    else:
        return None
