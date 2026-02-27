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

    def test_enu_to_ecef_prime_meridian_equator(self):
        # ENU (1, 0, 0)
        reference_point = Point(lat=0.0, lon=0.0, alt=0)
        reference_point_xyz = CoordinateTransformations.geodetic_to_cartesian(
            *reference_point.as_tuple()
        )
        east = 1.0
        north = 0.0
        up = 0.0

        expected_ecef = (
            reference_point_xyz[0],
            reference_point_xyz[1] + 1,
            reference_point_xyz[2],
        )
        calculated = CoordinateTransformations.enu_to_ecef(
            reference_point,
            (east, north, up),
        )
        self.assertEqual(expected_ecef, calculated)

        # ENU (0, 1, 0)
        reference_point = Point(lat=0.0, lon=0.0, alt=0)
        reference_point_xyz = CoordinateTransformations.geodetic_to_cartesian(
            *reference_point.as_tuple()
        )
        east = 0.0
        north = 1.0
        up = 0.0

        expected_ecef = (
            reference_point_xyz[0],
            reference_point_xyz[1],
            reference_point_xyz[2] + 1,
        )
        calculated = CoordinateTransformations.enu_to_ecef(
            reference_point,
            (east, north, up),
        )
        self.assertEqual(expected_ecef, calculated)

        # ENU (0, 0, 1)
        reference_point = Point(lat=0.0, lon=0.0, alt=0)
        reference_point_xyz = CoordinateTransformations.geodetic_to_cartesian(
            *reference_point.as_tuple()
        )
        east = 0.0
        north = 0.0
        up = 1.0

        expected_ecef = (
            reference_point_xyz[0] + 1,
            reference_point_xyz[1],
            reference_point_xyz[2],
        )
        calculated = CoordinateTransformations.enu_to_ecef(
            reference_point,
            (east, north, up),
        )
        self.assertEqual(expected_ecef, calculated)

    def test_enu_to_ecef_lon90_equator(self):
        # ENU (1, 0, 0)
        reference_point = Point(lat=0.0, lon=90.0, alt=0)
        reference_point_xyz = CoordinateTransformations.geodetic_to_cartesian(
            *reference_point.as_tuple()
        )
        east = 1.0
        north = 0.0
        up = 0.0

        expected_ecef = (
            reference_point_xyz[0] - 1,
            reference_point_xyz[1],
            reference_point_xyz[2],
        )
        calculated = CoordinateTransformations.enu_to_ecef(
            reference_point,
            (east, north, up),
        )
        self.assertEqual(expected_ecef, calculated)

        # ENU (0, 1, 0)
        reference_point = Point(lat=0.0, lon=90.0, alt=0)
        reference_point_xyz = CoordinateTransformations.geodetic_to_cartesian(
            *reference_point.as_tuple()
        )
        east = 0.0
        north = 1.0
        up = 0.0

        expected_ecef = (
            reference_point_xyz[0],
            reference_point_xyz[1],
            reference_point_xyz[2] + 1,
        )
        calculated = CoordinateTransformations.enu_to_ecef(
            reference_point,
            (east, north, up),
        )
        self.assertEqual(expected_ecef, calculated)

        # ENU (0, 0, 1)
        reference_point = Point(lat=0.0, lon=90.0, alt=0)
        reference_point_xyz = CoordinateTransformations.geodetic_to_cartesian(
            *reference_point.as_tuple()
        )
        east = 0.0
        north = 0.0
        up = 1.0

        expected_ecef = (
            reference_point_xyz[0],
            reference_point_xyz[1] + 1,
            reference_point_xyz[2],
        )
        calculated = CoordinateTransformations.enu_to_ecef(
            reference_point,
            (east, north, up),
        )
        # Here, absolute differences fail due to numerical precision issues.
        self.assertTrue(
            np.isclose(calculated, expected_ecef, atol=CONSISTENCY_DELTA).all()
        )

    def test_enu_to_ecef_consistency(self):
        rng = np.random.Generator(np.random.PCG64(seed=203907))

        for _ in range(100):
            reference_point = Point(
                lat=rng.uniform(-90.0, 90.0),
                lon=rng.uniform(-180.0, 180.0),
                alt=rng.uniform(0.0, 10_000.0),
            )

            point_enu = (
                rng.uniform(-10_000.0, 10_000.0),
                rng.uniform(-10_000.0, 10_000.0),
                rng.uniform(-10_000.0, 10_000.0),
            )
            point_ecef = CoordinateTransformations.enu_to_ecef(
                reference_point,
                point_enu,
            )
            point_enu_transformed = CoordinateTransformations.ecef_to_enu(
                reference_point,
                point_ecef,
            )
            # Here, absolute differences fail due to numerical precision issues.
            self.assertTrue(
                np.isclose(
                    point_enu_transformed, point_enu, atol=CONSISTENCY_DELTA
                ).all()
            )

    def test_ecef_to_enu_consistency(self):
        rng = np.random.Generator(np.random.PCG64(seed=203907))

        for _ in range(100):
            reference_point = Point(
                lat=rng.uniform(-90.0, 90.0),
                lon=rng.uniform(-180.0, 180.0),
                alt=rng.uniform(0.0, 10_000.0),
            )

            point = Point(
                lat=rng.uniform(-90.0, 90.0),
                lon=rng.uniform(-180.0, 180.0),
                alt=rng.uniform(0.0, 10_000.0),
            )
            point_ecef = CoordinateTransformations.geodetic_to_cartesian(
                *point.as_tuple()
            )
            point_enu = CoordinateTransformations.ecef_to_enu(
                reference_point,
                point_ecef,
            )
            point_ecef_transformed = CoordinateTransformations.enu_to_ecef(
                reference_point,
                point_enu,
            )
            # Here, absolute differences fail due to numerical precision issues.
            self.assertTrue(
                np.isclose(
                    point_ecef_transformed, point_ecef, atol=CONSISTENCY_DELTA
                ).all()
            )


if __name__ == "__main__":
    unittest.main()
