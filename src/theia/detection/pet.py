import datetime
import numpy as np
import pydantic
import scipy.constants as sc
from scipy.ndimage import binary_erosion

import theia
import theia.coordinates
from theia.detection.active import calculate_probability_of_detection
from theia.distance import line_of_sight_distance
from theia.grids import LatLonTerrainGrid
import theia.snr
from theia.types import PetDetection, PetSensor, Point, Target
from theia.util import get_clear_sky_attenuation


class PetDetector(pydantic.BaseModel):
    terrain_model: theia.TerrainModel
    rf_loss: float = theia.config.RF_LOSS

    def calculate_pet_detection(
        self,
        sensor: PetSensor,
        target: Target,
        rng: np.random.Generator,
    ) -> PetDetection | None:
        """
        Calculate a PET detection or return None if the SNR is not good enough,
        there is no line-of-sight or the target is not within the receiver's
        field-of-view.

        Parameters
        ----------
        sensor: PetSensor
            Sensor to use for the detection
        target: Target
            Target to be detected
        rng: np.random.Generator
            Random number generator

        Returns
        -------
        PetDetection | None
            Detection or ``None`` if the SNR is not high enough or there is no
            line-of-sight
        """
        los_ok = self.terrain_model.has_line_of_sight(
            sensor.transmitter.point,
            sensor.receiver.point,
        )

        if not los_ok:
            return

        dist = line_of_sight_distance(
            *sensor.receiver.point.as_tuple(),
            *sensor.transmitter.point.as_tuple(),
        )

        wavelength = sc.speed_of_light / (sensor.transmitter.frequency * 1e6)

        # We can "deactivate" some terms by setting them to one because the SNR
        # formula is multiplicative.
        snr_dB = theia.snr.calculate_snr(
            wavelength=wavelength,
            antenna_gain_transmitter=sensor.transmitter.antenna_gain,
            antenna_gain_receiver=sensor.receiver.antenna_gain(
                sensor.transmitter.frequency
            ),
            # RCS does not apply, so we set it to one since it is multiplicative.
            radar_cross_section=1.0,
            distance_transmitter_target=1.0,
            distance_receiver_target=dist,
            transmission_power=sensor.transmitter.power,
            bandwidth=sensor.transmitter.bandwidth,
            cpi_pulses=1,
            equivalent_temperature=sensor.receiver.noise_temperature,
            L_t=self.rf_loss,
            L_r=sensor.receiver.noise_figure,
            L_a=get_clear_sky_attenuation(sensor.transmitter.frequency) * dist / 1000.0,
            # TODO: Should we include these factors?
            polarization_factor=0.0,
            pattern_propagation_factor_receiver=0.0,
            pattern_propagation_factor_transmitter=0.0,
            is_one_way=True,
        )

        p = calculate_probability_of_detection(snr_dB, sensor.receiver.pfa)

        if rng.uniform(low=0, high=1) <= p:
            elev = theia.coordinates.calculate_elevation_angle(
                sensor.receiver.point,
                target.point,
            )
            azi = theia.coordinates.calculate_azimuth_angle(
                sensor.receiver.point,
                target.point,
            )

            if not (
                (sensor.receiver.min_elevation <= elev <= sensor.receiver.max_elevation)
                and (sensor.receiver.min_azimuth <= azi <= sensor.receiver.max_azimuth)
            ):
                return None

            sigma_elev = sensor.error_model.calculate_elevation_uncertainty(
                sensor,
                snr_dB,
            )
            sigma_azi = sensor.error_model.calculate_azimuth_uncertainty(
                sensor,
                snr_dB,
            )
            elev += rng.normal(0, sigma_elev)
            azi += rng.normal(0, sigma_azi)
            return PetDetection(
                detection_id=-1,
                time=datetime.datetime.fromtimestamp(0),
                sensor=sensor,
                target=target,
                elevation=elev,
                azimuth=azi,
                sigma_elevation=sigma_elev,
                sigma_azimuth=sigma_azi,
            )
        else:
            return None


def suggest_pet_receiver_locations(
    grid: LatLonTerrainGrid,
    target_positions: list[Point],
) -> list[Point]:
    all_has_los = np.full(
        (grid.n_points[0], grid.n_points[1], len(target_positions)),
        False,
        dtype=bool,
    )

    for i, target_pos in enumerate(target_positions):
        points = grid.points
        is_local_max = grid.is_local_maximum
        has_los = np.full(grid.n_points[0] * grid.n_points[1] * grid.n_points[2], False)
        for j, (point, is_max) in enumerate(
            zip(grid.points, is_local_max, strict=True)
        ):
            if not is_max:
                continue
            has_los[j] = grid.terrain_model.has_line_of_sight(
                target_pos,
                Point(lat=point[0], lon=point[1], alt=point[2]),
                60,
            )
        assert grid.n_points[2] == 1
        has_los = has_los.reshape(grid.n_points)[:, :, 0]
        has_los = np.logical_xor(has_los, binary_erosion(has_los))
        all_has_los[:, :, i] = has_los

    n_los = np.sum(all_has_los, axis=2)
    best_points = points[np.where(n_los.flatten() == np.max(n_los))]
    return best_points
