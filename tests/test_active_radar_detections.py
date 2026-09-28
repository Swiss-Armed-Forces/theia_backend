import math
import unittest
from unittest.mock import patch

import numpy as np

from theia.config import SIDC
from theia.coordinates import CoordinateTransformations, EcefToEnuTransformer
from theia.detection.active import calculate_monostatic_detection
from theia.terrain import DummyTerrain
from theia.test_data import build_flores_monostatic_radar
from theia.types import (
    ConstantRcsModel,
    MonostaticSensor,
    Point,
    Target,
    Velocity,
)

RADAR_POINT = Point(lat=46.95, lon=7.45, alt=550.0)
TARGET_RANGE = 20_000.0  # m


# Mocks: Make the target always detectable, so that only the field-of-view
# check decides whether a detection is returned.
def _snr_100dB(*args, **kwargs) -> float:
    return 100.0


def _doppler_1kHz(*args, **kwargs) -> float:
    return 1_000.0


def _radar(
    min_elevation: float = math.radians(-90.0),
    max_elevation: float = math.radians(90.0),
    min_azimuth: float = math.radians(-180.0),
    max_azimuth: float = math.radians(180.0),
) -> MonostaticSensor:
    """Build a radar whose receiver field-of-view is given in radians."""
    radar = build_flores_monostatic_radar(RADAR_POINT, 0, 1, 2)
    receiver = radar.receiver.model_copy(
        update=dict(
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            min_azimuth=min_azimuth,
            max_azimuth=max_azimuth,
        )
    )
    return radar.model_copy(update=dict(receiver=receiver))


def _target(azimuth_deg: float, elevation_deg: float) -> Target:
    """Place a target at the given azimuth/elevation as seen from the radar."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    east = TARGET_RANGE * math.cos(el) * math.sin(az)
    north = TARGET_RANGE * math.cos(el) * math.cos(az)
    up = TARGET_RANGE * math.sin(el)
    ecef = EcefToEnuTransformer(RADAR_POINT).enu_to_ecef((east, north, up))
    lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(*ecef)
    return Target(
        id=0,
        is_stationary=False,
        sidc=SIDC.UNKNOWN,
        point=Point(lat=lat, lon=lon, alt=alt),
        cross_section_model=ConstantRcsModel(rcs=1.0),
        velocity=Velocity(vx=300.0, vy=0.0, vz=0.0),
    )


@patch("theia.detection.active.calculate_doppler_shift", _doppler_1kHz)
@patch("theia.detection.active.calculate_monostatic_snr", _snr_100dB)
class MonostaticFieldOfViewTest(unittest.TestCase):
    """
    The receiver's field-of-view is stored in radians.

    Regression tests: The field-of-view check used to interpret it in degrees,
    which silently discarded detections e.g. at azimuths beyond 180°.
    """

    def _detect(self, radar: MonostaticSensor, target: Target):
        return calculate_monostatic_detection(
            DummyTerrain(has_los=True),
            radar,
            target,
            np.random.default_rng(seed=0),
        )

    def assertDetected(self, radar, azimuth_deg, elevation_deg):
        detection = self._detect(radar, _target(azimuth_deg, elevation_deg))
        self.assertIsNotNone(
            detection,
            f"Target at az={azimuth_deg}°, el={elevation_deg}° should be detected",
        )
        return detection

    def assertNotDetected(self, radar, azimuth_deg, elevation_deg):
        self.assertIsNone(
            self._detect(radar, _target(azimuth_deg, elevation_deg)),
            f"Target at az={azimuth_deg}°, el={elevation_deg}° should not be detected",
        )

    def test_full_field_of_view_detects_all_azimuths(self):
        radar = _radar()
        for azimuth_deg in range(0, 360, 15):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertDetected(radar, azimuth_deg, 10.0)

    def test_full_field_of_view_detects_all_elevations(self):
        radar = _radar()
        for elevation_deg in (-60.0, -10.0, 0.0, 10.0, 45.0, 80.0):
            with self.subTest(elevation_deg=elevation_deg):
                self.assertDetected(radar, 200.0, elevation_deg)

    def test_detection_angles_match_target_position(self):
        detection = self.assertDetected(_radar(), 270.0, 20.0)
        self.assertAlmostEqual(detection.azimuth_angle, math.radians(270.0), places=3)
        self.assertAlmostEqual(detection.elevation_angle, math.radians(20.0), places=3)

    def test_azimuth_sector_across_north(self):
        # Sector [-90°, +90°], i.e. the northern half.
        radar = _radar(min_azimuth=math.radians(-90), max_azimuth=math.radians(90))
        for azimuth_deg in (0.0, 45.0, 85.0, 275.0, 315.0):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertDetected(radar, azimuth_deg, 10.0)
        for azimuth_deg in (95.0, 180.0, 265.0):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertNotDetected(radar, azimuth_deg, 10.0)

    def test_azimuth_sector_south_west(self):
        # Sector [180°, 270°]
        radar = _radar(min_azimuth=math.radians(180), max_azimuth=math.radians(270))
        for azimuth_deg in (185.0, 225.0, 265.0):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertDetected(radar, azimuth_deg, 10.0)
        for azimuth_deg in (0.0, 90.0, 175.0, 275.0):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertNotDetected(radar, azimuth_deg, 10.0)

    def test_elevation_limits(self):
        radar = _radar(
            min_elevation=math.radians(0.0),
            max_elevation=math.radians(30.0),
        )
        for elevation_deg in (1.0, 15.0, 29.0):
            with self.subTest(elevation_deg=elevation_deg):
                self.assertDetected(radar, 90.0, elevation_deg)
        for elevation_deg in (-10.0, 31.0, 60.0):
            with self.subTest(elevation_deg=elevation_deg):
                self.assertNotDetected(radar, 90.0, elevation_deg)


if __name__ == "__main__":
    unittest.main()
