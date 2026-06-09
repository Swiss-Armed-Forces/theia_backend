import unittest

import numpy as np

from theia.coordinates import CoordinateTransformations, EcefToEnuTransformer
from theia.types import Point, Velocity

CONSISTENCY_DELTA = 1e-12


class CoordinateTransformationTest(unittest.TestCase):
    def test_cartesian_to_geodetic(self):
        rng = np.random.Generator(np.random.PCG64(seed=4987897849))
        for _ in range(10_000):
            lat = rng.uniform(-90, 90)
            lon = rng.uniform(-180, 180)
            alt = rng.uniform(0, 10_000)

            x, y, z = CoordinateTransformations.geodetic_to_cartesian(lat, lon, alt)
            lat_new, lon_new, alt_new = CoordinateTransformations.cartesian_to_geodetic(
                x,
                y,
                z,
            )

            self.assertAlmostEqual(lat, lat_new, delta=1e-10)
            self.assertAlmostEqual(lon, lon_new, delta=CONSISTENCY_DELTA)
            self.assertAlmostEqual(alt, alt_new, delta=1e-5)
        
    def test_geodetic_to_cartesian_against_matlab_reference(self):
        """
        Validate ecef_to_enu against Octave's ecef2enu reference output.

        Octave call:
            pkg load mapping
            output_precision(10)
            lat = 48.8566;
            lon =  2.3522;
            alt = 35.0;

            [x, y, z] = geodetic2ecef(referenceEllipsoid('wgs84'), lat, lon, alt)
        Octave output:
            x = 4200937.804
            y = 172560.7214
            z = 4780107.699
        """
        # Local ENU origin (satellite position in geodetic coordinates)
        point = Point(lat=48.8566, lon=2.3522, alt=35.0)
        x, y, z = CoordinateTransformations.geodetic_to_cartesian(*point.as_tuple())

        # Octave reference values.
        self.assertAlmostEqual(x, 4200937.804, delta=0.001)
        self.assertAlmostEqual(y, 172560.7214, delta=0.001)
        self.assertAlmostEqual(z, 4780107.699, delta=0.001)

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
        transformer = EcefToEnuTransformer(reference_point)
        calculated = transformer.enu_to_ecef((east, north, up))
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
        transformer = EcefToEnuTransformer(reference_point)
        calculated = transformer.enu_to_ecef((east, north, up))
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
        transformer = EcefToEnuTransformer(reference_point)
        calculated = transformer.enu_to_ecef((east, north, up))
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
        transformer = EcefToEnuTransformer(reference_point)
        calculated = transformer.enu_to_ecef((east, north, up))
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
        transformer = EcefToEnuTransformer(reference_point)
        calculated = transformer.enu_to_ecef((east, north, up))
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
        transformer = EcefToEnuTransformer(reference_point)
        calculated = transformer.enu_to_ecef((east, north, up))
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
            transformer = EcefToEnuTransformer(reference_point)
            point_ecef = transformer.enu_to_ecef(point_enu)
            transformer = EcefToEnuTransformer(reference_point)
            point_enu_transformed = transformer.ecef_to_enu(point_ecef)
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
            transformer = EcefToEnuTransformer(reference_point)
            point_enu = transformer.ecef_to_enu(point_ecef)
            point_ecef_transformed = transformer.enu_to_ecef(point_enu)
            # Here, absolute differences fail due to numerical precision issues.
            self.assertTrue(
                np.isclose(
                    point_ecef_transformed, point_ecef, atol=CONSISTENCY_DELTA
                ).all()
            )

    def test_ecef_to_enu_prime_meridian_equator(self):
        """
        At lat=0, lon=0 the reference point lies on the +X axis.
        Unit ECEF displacements must map to the expected ENU basis vectors:
          +X → Up (0, 0, 1)
          +Y → East (1, 0, 0)
          +Z → North (0, 1, 0)
        """
        reference_point = Point(lat=0.0, lon=0.0, alt=0.0)
        cx, cy, cz = CoordinateTransformations.geodetic_to_cartesian(
            *reference_point.as_tuple()
        )
        transformer = EcefToEnuTransformer(reference_point)

        east, north, up = transformer.ecef_to_enu((cx + 1, cy, cz))
        self.assertAlmostEqual(east, 0.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(north, 0.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(up, 1.0, delta=CONSISTENCY_DELTA)

        east, north, up = transformer.ecef_to_enu((cx, cy + 1, cz))
        self.assertAlmostEqual(east, 1.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(north, 0.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(up, 0.0, delta=CONSISTENCY_DELTA)

        east, north, up = transformer.ecef_to_enu((cx, cy, cz + 1))
        self.assertAlmostEqual(east, 0.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(north, 1.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(up, 0.0, delta=CONSISTENCY_DELTA)

    def test_ecef_to_enu_lon90_equator(self):
        """
        At lat=0, lon=90 the reference point lies on the +Y axis.
        Unit ECEF displacements must map to the expected ENU basis vectors:
          -X → East (1, 0, 0)
          +Y → Up (0, 0, 1)
          +Z → North (0, 1, 0)
        """
        reference_point = Point(lat=0.0, lon=90.0, alt=0.0)
        cx, cy, cz = CoordinateTransformations.geodetic_to_cartesian(
            *reference_point.as_tuple()
        )
        transformer = EcefToEnuTransformer(reference_point)

        east, north, up = transformer.ecef_to_enu((cx - 1, cy, cz))
        self.assertAlmostEqual(east, 1.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(north, 0.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(up, 0.0, delta=CONSISTENCY_DELTA)

        east, north, up = transformer.ecef_to_enu((cx, cy + 1, cz))
        self.assertAlmostEqual(east, 0.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(north, 0.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(up, 1.0, delta=CONSISTENCY_DELTA)

        east, north, up = transformer.ecef_to_enu((cx, cy, cz + 1))
        self.assertAlmostEqual(east, 0.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(north, 1.0, delta=CONSISTENCY_DELTA)
        self.assertAlmostEqual(up, 0.0, delta=CONSISTENCY_DELTA)

    def test_ecef_to_enu_reference_point_is_origin(self):
        """The reference point itself must transform to ENU (0, 0, 0)."""
        rng = np.random.Generator(np.random.PCG64(seed=111))
        for _ in range(20):
            reference_point = Point(
                lat=rng.uniform(-90.0, 90.0),
                lon=rng.uniform(-180.0, 180.0),
                alt=rng.uniform(0.0, 10_000.0),
            )
            ref_ecef = CoordinateTransformations.geodetic_to_cartesian(
                *reference_point.as_tuple()
            )
            transformer = EcefToEnuTransformer(reference_point)
            east, north, up = transformer.ecef_to_enu(ref_ecef)
            self.assertAlmostEqual(east, 0.0, delta=CONSISTENCY_DELTA)
            self.assertAlmostEqual(north, 0.0, delta=CONSISTENCY_DELTA)
            self.assertAlmostEqual(up, 0.0, delta=CONSISTENCY_DELTA)

    def test_ecef_to_enu_physical_directions(self):
        """
        For a mid-latitude reference point, verify that:
        - a target at slightly higher latitude gives positive North, ~zero East
        - a target at slightly higher longitude gives positive East, ~zero North
        - a target at higher altitude gives positive Up, ~zero East and North
        """
        reference = Point(lat=45.0, lon=10.0, alt=100.0)
        transformer = EcefToEnuTransformer(reference)

        target_north_ecef = CoordinateTransformations.geodetic_to_cartesian(
            *Point(lat=45.001, lon=10.0, alt=100.0).as_tuple()
        )
        east, north, up = transformer.ecef_to_enu(target_north_ecef)
        self.assertGreater(north, 0.0)
        self.assertAlmostEqual(east, 0.0, delta=1e-3)

        target_east_ecef = CoordinateTransformations.geodetic_to_cartesian(
            *Point(lat=45.0, lon=10.001, alt=100.0).as_tuple()
        )
        east, north, up = transformer.ecef_to_enu(target_east_ecef)
        self.assertGreater(east, 0.0)
        self.assertAlmostEqual(north, 0.0, delta=1e-3)

        target_up_ecef = CoordinateTransformations.geodetic_to_cartesian(
            *Point(lat=45.0, lon=10.0, alt=200.0).as_tuple()
        )
        east, north, up = transformer.ecef_to_enu(target_up_ecef)
        self.assertGreater(up, 0.0)
        self.assertAlmostEqual(east, 0.0, delta=1e-3)
        self.assertAlmostEqual(north, 0.0, delta=1e-3)

    def test_ecef_to_enu_against_matlab_reference(self):
        """
        Validate ecef_to_enu against Octave's ecef2enu reference output.

        Reference values taken verbatim from the Matlab ecef2enu documentation
        example (wgs84Ellipsoid with length unit 'kilometer').  All values are
        converted to metres for this test.

        Octave call:
            pkg load mapping
            output_precision(10)
            wgs84 = wgs84Ellipsoid('kilometer');
            [xEast, yNorth, zUp] = ecef2enu(5507.5289, 4556.2241, 6012.8208, 45.9132, 36.7484, 1877.7532, wgs84)
        Octave output:
            xEast = 355.6012615
            yNorth = -923.0831559
            zUp = 1041.016424
        """
        # Local ENU origin (satellite position in geodetic coordinates)
        reference_point = Point(lat=45.9132, lon=36.7484, alt=1877753.2)

        # ECEF position of orbital debris [m]
        debris_ecef = (5507528.9, 4556224.1, 6012820.8)

        transformer = EcefToEnuTransformer(reference_point)
        east, north, up = transformer.ecef_to_enu(debris_ecef)

        # Octave reference values converted to metres.
        self.assertAlmostEqual(east, 355601.2615, delta=0.001)
        self.assertAlmostEqual(north, -923083.1559, delta=0.001)
        self.assertAlmostEqual(up, 1041016.424, delta=0.001)

    def test_ecef_to_enu_multiple_matches_single(self):
        """ecef_to_enu_multiple must return identical results to ecef_to_enu for each point."""
        rng = np.random.Generator(np.random.PCG64(seed=42))
        reference_point = Point(
            lat=rng.uniform(-90.0, 90.0),
            lon=rng.uniform(-180.0, 180.0),
            alt=rng.uniform(0.0, 10_000.0),
        )
        transformer = EcefToEnuTransformer(reference_point)

        points_ecef = [
            CoordinateTransformations.geodetic_to_cartesian(
                rng.uniform(-90.0, 90.0),
                rng.uniform(-180.0, 180.0),
                rng.uniform(0.0, 10_000.0),
            )
            for _ in range(20)
        ]

        results_single = np.array([transformer.ecef_to_enu(p) for p in points_ecef])
        results_multiple = transformer.ecef_to_enu_multiple(points_ecef)

        self.assertTrue(
            np.isclose(results_single, results_multiple, atol=CONSISTENCY_DELTA).all()
        )


if __name__ == "__main__":
    unittest.main()
