import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import kstest, norm

from theia.config import SIDC_UNKNOWN
from theia.detection.pet import PetDetector
from theia.terrain import DummyTerrain
from theia.test_data import build_airborne_pet_receiver, build_flores_monostatic_radar
from theia.types import (
    ConstantRcsModel,
    PetDetection,
    PetMeasurementModel,
    PetSensor,
    Point,
    Target,
    Velocity,
)


# Mocks.
def _snr_100dB(*args, **kwargs) -> float:
    """Fake SNR calculation."""
    return 100.0


def _snr_0dB(*args, **kwargs) -> float:
    """Fake SNR calculation."""
    return 0.0


def _target(point: Point) -> Target:
    return Target(
        id=0,
        is_stationary=False,
        sidc=SIDC_UNKNOWN,
        point=point,
        cross_section_model=ConstantRcsModel(rcs=1.0),
        velocity=Velocity(vx=300.0, vy=0.0, vz=0.0),
    )


def _calculate_elevation_in_cone(
    transmitter_point: Point, target_point: Point
) -> float:
    return math.radians(20)


def _calculate_elevation_out_of_cone(
    transmitter_point: Point, target_point: Point
) -> float:
    return math.radians(40)


def _calculate_azimuth(transmitter_point: Point, target_point: Point) -> float:
    return math.radians(45)


class PetTest(unittest.TestCase):
    def setUp(self):
        p_rx = Point(lat=0, lon=0, alt=0)
        p_tgt = Point(lat=10, lon=10, alt=1000)

        rx = build_airborne_pet_receiver(
            0,
            p_rx,
        )
        tx = build_flores_monostatic_radar(
            p_tgt,
            sensor_id=0,
            rx_id=1,
            tx_id=0,
        ).transmitter
        self._certain_pet_sensor = PetSensor(
            id=0,
            transmitter=tx,
            receiver=rx,
            target=_target(p_tgt),
            error_model=PetMeasurementModel(
                min_elevation_uncertainty=0.0,
                max_elevation_uncertainty=0.0,
                min_azimuth_uncertainty=0.0,
                max_azimuth_uncertainty=0.0,
            ),
        )
        self._pet_sensor = PetSensor(
            id=0,
            transmitter=tx,
            receiver=rx,
            target=_target(p_tgt),
            error_model=PetMeasurementModel(),
        )
        self._rng = np.random.default_rng(seed=9207998)

    def test_no_los(self):
        """There must not be a detection if there is no LOS."""
        terrain = DummyTerrain(has_los=False)
        detector = PetDetector(terrain_model=terrain)
        result = detector.calculate_pet_detection(
            self._pet_sensor,
            _target(Point(lat=10, lon=10, alt=1000)),
            self._rng,
        )
        self.assertIsNone(result)

    @patch(
        "theia.coordinates.calculate_elevation_angle", _calculate_elevation_out_of_cone
    )
    @patch("theia.coordinates.calculate_azimuth_angle", _calculate_azimuth)
    def test_not_in_fov(self):
        """
        There must not be a detection if the receiver has LOS, but the target
        is outside the receiver's field-of-view.
        """
        terrain = DummyTerrain(has_los=True)
        detector = PetDetector(terrain_model=terrain)

        result = detector.calculate_pet_detection(
            self._pet_sensor,
            _target(Point(lat=10, lon=10, alt=1000)),
            self._rng,
        )
        self.assertIsNone(result)

    @patch("theia.coordinates.calculate_elevation_angle", _calculate_elevation_in_cone)
    @patch("theia.coordinates.calculate_azimuth_angle", _calculate_azimuth)
    @patch("theia.snr.calculate_snr", _snr_0dB)
    def test_too_low_snr(self):
        """There must not be a detection if the SNR is too low."""
        terrain = DummyTerrain(has_los=True)
        detector = PetDetector(terrain_model=terrain)

        result = detector.calculate_pet_detection(
            self._pet_sensor,
            _target(Point(lat=10, lon=10, alt=1000)),
            self._rng,
        )
        self.assertIsNone(result)

    @patch("theia.coordinates.calculate_elevation_angle", _calculate_elevation_in_cone)
    @patch("theia.coordinates.calculate_azimuth_angle", _calculate_azimuth)
    @patch("theia.snr.calculate_snr", _snr_100dB)
    def test_success(self):
        """There must be a detection if there is a line of sight."""
        terrain = DummyTerrain(has_los=True)
        detector = PetDetector(terrain_model=terrain)

        result = detector.calculate_pet_detection(
            self._pet_sensor,
            _target(Point(lat=10, lon=10, alt=1000)),
            self._rng,
        )
        self.assertIsNotNone(result)

    @patch("theia.coordinates.calculate_elevation_angle", _calculate_elevation_in_cone)
    @patch("theia.coordinates.calculate_azimuth_angle", _calculate_azimuth)
    @patch("theia.snr.calculate_snr", _snr_100dB)
    def test_measured_values_no_uncertainty(self):
        """Detections must have the correct measured values."""
        terrain = DummyTerrain(has_los=True)
        detector = PetDetector(terrain_model=terrain)

        result = detector.calculate_pet_detection(
            self._certain_pet_sensor,
            _target(Point(lat=10, lon=10, alt=1000)),
            self._rng,
        )
        self.assertAlmostEqual(result.elevation, math.radians(20))
        self.assertAlmostEqual(result.azimuth, math.radians(45))

    @patch("theia.coordinates.calculate_elevation_angle", _calculate_elevation_in_cone)
    @patch("theia.coordinates.calculate_azimuth_angle", _calculate_azimuth)
    @patch("theia.snr.calculate_snr", _snr_100dB)
    def test_uncertainty(self):
        """Detections must follow the probability of detection."""
        expected_sigma_elevation = (
            self._pet_sensor.error_model.calculate_elevation_uncertainty(
                self._pet_sensor,
                100.0,
            )
        )
        expected_sigma_azimuth = (
            self._pet_sensor.error_model.calculate_azimuth_uncertainty(
                self._pet_sensor,
                100.0,
            )
        )

        terrain = DummyTerrain(has_los=True)
        detector = PetDetector(terrain_model=terrain)

        results: list[PetDetection] = []
        for _ in range(10_000):
            results.append(
                detector.calculate_pet_detection(
                    self._pet_sensor,
                    _target(Point(lat=10, lon=10, alt=1000)),
                    self._rng,
                )
            )
        elevations = [result.elevation for result in results]
        azimuths = [result.azimuth for result in results]

        stat, p_value = kstest(
            elevations,
            cdf=norm(loc=math.radians(20), scale=expected_sigma_elevation).cdf,
        )
        self.assertGreaterEqual(p_value, 0.05)

        stat, p_value = kstest(
            azimuths,
            cdf=norm(loc=math.radians(45), scale=expected_sigma_azimuth).cdf,
        )
        self.assertGreaterEqual(p_value, 0.05)


if __name__ == "__main__":
    unittest.main()
