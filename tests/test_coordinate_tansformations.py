import unittest

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.types import Point, Velocity


class CoordinateTransformationTest(unittest.TestCase):
    def test_velocity_transformation_consistency_geodetic_cartesian_geodetic(self):
        rng = np.random.Generator(np.random.PCG64(seed=4987897849))
        for _ in range(100):
            v_latlonalt = rng.uniform(-600, 600, size=(3,))
            lon = rng.uniform(-np.pi / 2, np.pi / 2)
            lat = rng.uniform(-np.pi, np.pi)
            alt = rng.uniform(0, 10_000)
            p = Point(
                lat=lat,
                lon=lon,
                alt=alt,
            )
            v_xyz = CoordinateTransformations.velocity_geodetic_to_cartesian(
                p, *v_latlonalt
            )
            v_latlonalt_transformed = np.asarray(
                CoordinateTransformations.velocity_cartesian_to_geodetic(p, v_xyz)
            )

            self.assertAlmostEqual(
                v_latlonalt[0], v_latlonalt_transformed[0], delta=0.0015
            )
            self.assertAlmostEqual(
                v_latlonalt[1], v_latlonalt_transformed[1], delta=0.0015
            )
            self.assertAlmostEqual(
                v_latlonalt[2], v_latlonalt_transformed[2], delta=0.0015
            )

    def test_velocity_transformation_consistency_cartesian_geodetic_cartesian(self):
        rng = np.random.Generator(np.random.PCG64(seed=4987897849))
        for _ in range(100):
            v_xyz = rng.uniform(-600, 600, size=(3,))
            lon = rng.uniform(-np.pi / 2, np.pi / 2)
            lat = rng.uniform(-np.pi, np.pi)
            alt = rng.uniform(0, 10_000)
            p = Point(
                lat=lat,
                lon=lon,
                alt=alt,
            )
            velocity = Velocity(vx=v_xyz[0], vy=v_xyz[1], vz=v_xyz[2])
            v_latlonalt = CoordinateTransformations.velocity_cartesian_to_geodetic(
                p, velocity
            )
            v_xyz_transformed = np.asarray(
                CoordinateTransformations.velocity_geodetic_to_cartesian(
                    p,
                    *v_latlonalt,
                ).as_tuple()
            )

            self.assertAlmostEqual(v_xyz[0], v_xyz_transformed[0], delta=0.0015)
            self.assertAlmostEqual(v_xyz[1], v_xyz_transformed[1], delta=0.0015)
            self.assertAlmostEqual(v_xyz[2], v_xyz_transformed[2], delta=0.0015)

    def test_against_manual_at_equator_prime_meridian(self):
        p = Point(
            lat=0,
            lon=0,
            alt=0,
        )

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p,
            1.,
            0.,
            0.
        )
        self.assertAlmostEqual(v_latlonalt.vx, 0, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vy, 0, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vz, 1, delta=1e-6)

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p,
            0.,
            1.,
            0.
        )
        self.assertAlmostEqual(v_latlonalt.vx, 0, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vy, 1, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vz, 0, delta=1e-6)

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p,
            0.,
            0.,
            1.
        )
        self.assertAlmostEqual(v_latlonalt.vx, 1, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vy, 0, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vz, 0, delta=1e-6)
    
    def test_against_manual_at_equator_90E(self):
        p = Point(
            lat=0,
            lon=90,
            alt=0,
        )

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p,
            1.,
            0.,
            0.
        )
        self.assertAlmostEqual(v_latlonalt.vx, 0, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vy, 0, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vz, 1, delta=1e-6)

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p,
            0.,
            1.,
            0.
        )
        self.assertAlmostEqual(v_latlonalt.vx, -1, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vy, 0, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vz, 0, delta=1e-6)

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p,
            0.,
            0.,
            1.
        )
        self.assertAlmostEqual(v_latlonalt.vx, 0, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vy, 1, delta=1e-6)
        self.assertAlmostEqual(v_latlonalt.vz, 0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
