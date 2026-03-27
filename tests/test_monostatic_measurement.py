import unittest

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.measurement import MonostaticMeasurementTransformations
from theia.test_data import get_uetliberg_radar, sample_position

TOLERANCE_DEGREE = 1e-10
TOLERANCE_METER = 1e-5

class MonostaticMeasurementTest(unittest.TestCase):
    def test_perfect_measurement(self):
        # Test consistency of the measurement without uncertainty.
        rng = np.random.Generator(np.random.PCG64(seed=480792990))
        radar = get_uetliberg_radar()
        p_observer = radar.receiver.point
        for _ in range(1000):
            p_target = sample_position(
                rng, lat_min=40, lat_max=50, lon_min=0, lon_max=17
            )
            p_target_ecef = CoordinateTransformations.geodetic_to_cartesian(
                p_target.lat,
                p_target.lon,
                p_target.alt,
            )
            elevation, azimuth, target_range = (
                MonostaticMeasurementTransformations.cartesian_to_elevation_azimuth_range(
                    p_observer,
                    p_target_ecef,
                )
            )
            p_target_ecef_reconstructed = MonostaticMeasurementTransformations.elevation_azimuth_range_to_cartesian(
                p_observer,
                elevation,
                azimuth,
                target_range,
            )

            for i in range(3):
                self.assertAlmostEqual(
                    p_target_ecef[i],
                    p_target_ecef_reconstructed[i],
                    delta=1e-9,
                )

            lat_recon, lon_recon, alt_recon = (
                CoordinateTransformations.cartesian_to_geodetic(
                    *p_target_ecef_reconstructed
                )
            )

            self.assertAlmostEqual(
                p_target.lat,
                lat_recon,
                delta=TOLERANCE_DEGREE,
            )
            self.assertAlmostEqual(
                p_target.lon,
                lon_recon,
                delta=TOLERANCE_DEGREE,
            )
            self.assertAlmostEqual(
                p_target.alt,
                alt_recon,
                delta=TOLERANCE_METER,
            )


if __name__ == "__main__":
    unittest.main()
