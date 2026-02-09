import unittest

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.types import Point, Velocity

CONSISTENCY_DELTA = 1e-12


class CoordinateTransformationTest(unittest.TestCase):
    def test_velocity_transformation_consistency_geodetic_cartesian_geodetic(self):
        rng = np.random.Generator(np.random.PCG64(seed=4987897849))
        for _ in range(100):
            v_latlonalt = rng.uniform(-600, 600, size=(3,))
            lat = rng.uniform(-90, 90)
            lon = rng.uniform(-180, 180)
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
                v_latlonalt[0],
                v_latlonalt_transformed[0],
                delta=CONSISTENCY_DELTA,
            )
            self.assertAlmostEqual(
                v_latlonalt[1],
                v_latlonalt_transformed[1],
                delta=CONSISTENCY_DELTA,
            )
            self.assertAlmostEqual(
                v_latlonalt[2],
                v_latlonalt_transformed[2],
                delta=CONSISTENCY_DELTA,
            )

    def test_velocity_transformation_consistency_cartesian_geodetic_cartesian(self):
        rng = np.random.Generator(np.random.PCG64(seed=4987897849))
        for _ in range(100):
            v_xyz = rng.uniform(-600, 600, size=(3,))
            lat = rng.uniform(-90, 90)
            lon = rng.uniform(-180, 180)
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

            self.assertAlmostEqual(
                v_xyz[0],
                v_xyz_transformed[0],
                delta=CONSISTENCY_DELTA,
            )
            self.assertAlmostEqual(
                v_xyz[1],
                v_xyz_transformed[1],
                delta=CONSISTENCY_DELTA,
            )
            self.assertAlmostEqual(
                v_xyz[2],
                v_xyz_transformed[2],
                delta=CONSISTENCY_DELTA,
            )

    def test_against_manual_at_equator_prime_meridian(self):
        p = Point(
            lat=0,
            lon=0,
            alt=0,
        )

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p, 1.0, 0.0, 0.0
        )
        self.assertAlmostEqual(v_latlonalt.vx, 0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vy, 0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vz, 1, delta=CONSISTENCY_DELTA)

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p, 0.0, 1.0, 0.0
        )
        self.assertAlmostEqual(v_latlonalt.vx, 0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vy, 1, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vz, 0, delta=CONSISTENCY_DELTA)

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p, 0.0, 0.0, 1.0
        )
        self.assertAlmostEqual(v_latlonalt.vx, 1, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vy, 0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vz, 0, delta=CONSISTENCY_DELTA)

    def test_against_manual_at_equator_90E(self):
        p = Point(
            lat=0,
            lon=90,
            alt=0,
        )

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p, 1.0, 0.0, 0.0
        )
        self.assertAlmostEqual(v_latlonalt.vx, 0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vy, 0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vz, 1, delta=CONSISTENCY_DELTA)

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p, 0.0, 1.0, 0.0
        )
        self.assertAlmostEqual(v_latlonalt.vx, -1, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vy, 0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vz, 0, delta=CONSISTENCY_DELTA)

        v_latlonalt = CoordinateTransformations.velocity_geodetic_to_cartesian(
            p, 0.0, 0.0, 1.0
        )
        self.assertAlmostEqual(v_latlonalt.vx, 0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vy, 1, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(v_latlonalt.vz, 0, delta=CONSISTENCY_DELTA)

    def test_at_north_pole(self):
        """At pole, longitude direction is undefined; only alt is well-defined"""
        p = Point(lat=90, lon=0, alt=0)
        v = CoordinateTransformations.velocity_geodetic_to_cartesian(p, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(
            v.vz, 1, delta=CONSISTENCY_DELTA
        )  # Up = +Z at north pole

    def test_at_south_pole(self):
        p = Point(lat=-90, lon=0, alt=0)
        v = CoordinateTransformations.velocity_geodetic_to_cartesian(p, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(
            v.vz, -1, delta=CONSISTENCY_DELTA
        )  # Up = -Z at south pole

    def test_magnitude_preservation(self):
        """Velocity magnitude should be preserved in transformation"""
        rng = np.random.Generator(np.random.PCG64(seed=12345))
        for _ in range(50):
            # Define random position
            lon = rng.uniform(-90, 90)
            lat = rng.uniform(-180, 180)
            alt = rng.uniform(0, 10_000)
            p = Point(lat=lat, lon=lon, alt=alt)

            # Random velocity
            v_geo = rng.uniform(-100, 100, 3)
            mag_geo = np.linalg.norm(v_geo)

            v_cart = CoordinateTransformations.velocity_geodetic_to_cartesian(p, *v_geo)
            mag_cart = np.linalg.norm([v_cart.vx, v_cart.vy, v_cart.vz])

            self.assertAlmostEqual(mag_geo, mag_cart, delta=CONSISTENCY_DELTA)


if __name__ == "__main__":
    unittest.main()
