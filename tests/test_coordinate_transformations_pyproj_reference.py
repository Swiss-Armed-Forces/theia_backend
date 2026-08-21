"""
Reference tests for CoordinateTransformations against an independent pyproj Transformer.
"""

import unittest

import numpy as np
import pyproj

from theia.coordinates import CoordinateTransformations

# Independent reference transformer, deliberately not shared with the
# implementation under test.
_REFERENCE_TRANSFORMER = pyproj.Transformer.from_proj(
    pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84"),
    pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84"),
)

N_SAMPLES = 10_000
SEED = 20260821

# Absolute tolerances. ECEF coordinates are on the order of 1e6-1e7 m, so 1e-6 m
# (1 micrometer) absolute agreement with pyproj is a strict but achievable bar
# for a correct closed-form implementation operating in float64.
XYZ_ATOL_M = 1e-6
LAT_LON_ATOL_DEG = 1e-9

# ALT_ATOL_M is looser than XYZ_ATOL_M/LAT_LON_ATOL_DEG on purpose. The
# cartesian_to_geodetic tests below don't compare against a true/independent
# altitude -- they compare against _REFERENCE_TRANSFORMER's own INVERSE
# transform of a point, i.e. pyproj's own round trip. Measuring that round
# trip directly (forward geodetic->ECEF, then inverse ECEF->geodetic, both via
# pyproj, compared to the original altitude) shows pyproj's own inverse
# geocentric transform is *not* accurate to 1e-6 m: for the altitude range
# used by test_random_points_global (-1_000 m to 50_000 m) its own round-trip
# altitude error reaches ~2.6e-5 m. A hand-verified closed-form implementation
# can end up *more* accurate than pyproj's own inverse, at
# which point it necessarily disagrees with pyproj's (slightly off) answer by
# something on the order of pyproj's own error, so the tolerance has to
# accommodate pyproj's imprecision, not just float64 rounding.
ALT_ATOL_M = 5e-5

# Looser tolerances specifically for test_negative_and_high_altitude's
# extended altitude range (-10_000 m to 1_000_000 m). pyproj's own round-trip
# error grows with altitude: at up to 1_000_000 m it reaches ~7.4e-3 m in
# altitude and ~5.0e-8 deg in latitude (measured the same way as ALT_ATOL_M
# above, over 500 samples with SEED + 1) -- both well past the tolerances
# used elsewhere in this file. These constants carry roughly a 2-4x margin
# over that measured worst case.
HIGH_ALTITUDE_LAT_LON_ATOL_DEG = 2e-7
HIGH_ALTITUDE_ALT_ATOL_M = 2e-2


def _sample_geodetic_points(
    n: int,
    rng: np.random.Generator,
    lat_range: tuple[float, float] = (-90, 90),
    alt_range: tuple[float, float] = (-1_000, 50_000),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = rng.uniform(*lat_range, size=n)
    lon = rng.uniform(-180, 180, size=n)
    alt = rng.uniform(*alt_range, size=n)
    return lat, lon, alt


class GeodeticToCartesianVsPyprojTest(unittest.TestCase):
    """geodetic_to_cartesian must agree with pyproj to (near) floating point precision."""

    def test_random_points_global(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED))
        lat, lon, alt = _sample_geodetic_points(N_SAMPLES, rng)

        x_ref, y_ref, z_ref = _REFERENCE_TRANSFORMER.transform(
            lon, lat, alt, radians=False
        )

        for i in range(N_SAMPLES):
            x, y, z = CoordinateTransformations.geodetic_to_cartesian(
                float(lat[i]), float(lon[i]), float(alt[i])
            )
            self.assertAlmostEqual(x, x_ref[i], delta=XYZ_ATOL_M)
            self.assertAlmostEqual(y, y_ref[i], delta=XYZ_ATOL_M)
            self.assertAlmostEqual(z, z_ref[i], delta=XYZ_ATOL_M)

    def test_return_type_is_python_float(self) -> None:
        x, y, z = CoordinateTransformations.geodetic_to_cartesian(46.8, 8.2, 500.0)
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertIsInstance(z, float)

    def test_poles(self) -> None:
        for lat in (90.0, -90.0):
            for lon in (0.0, 45.0, 179.9, -179.9):
                x, y, z = CoordinateTransformations.geodetic_to_cartesian(lat, lon, 0.0)
                x_ref, y_ref, z_ref = _REFERENCE_TRANSFORMER.transform(
                    lon, lat, 0.0, radians=False
                )
                self.assertAlmostEqual(x, x_ref, delta=XYZ_ATOL_M)
                self.assertAlmostEqual(y, y_ref, delta=XYZ_ATOL_M)
                self.assertAlmostEqual(z, z_ref, delta=XYZ_ATOL_M)

    def test_equator_and_antimeridian(self) -> None:
        for lat, lon in [(0.0, 0.0), (0.0, 180.0), (0.0, -180.0), (0.0, 90.0)]:
            x, y, z = CoordinateTransformations.geodetic_to_cartesian(lat, lon, 0.0)
            x_ref, y_ref, z_ref = _REFERENCE_TRANSFORMER.transform(
                lon, lat, 0.0, radians=False
            )
            self.assertAlmostEqual(x, x_ref, delta=XYZ_ATOL_M)
            self.assertAlmostEqual(y, y_ref, delta=XYZ_ATOL_M)
            self.assertAlmostEqual(z, z_ref, delta=XYZ_ATOL_M)

    def test_negative_and_high_altitude(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 1))
        lat, lon, alt = _sample_geodetic_points(
            500, rng, alt_range=(-10_000, 1_000_000)
        )
        x_ref, y_ref, z_ref = _REFERENCE_TRANSFORMER.transform(
            lon, lat, alt, radians=False
        )
        for i in range(500):
            x, y, z = CoordinateTransformations.geodetic_to_cartesian(
                float(lat[i]), float(lon[i]), float(alt[i])
            )
            self.assertAlmostEqual(x, x_ref[i], delta=XYZ_ATOL_M)
            self.assertAlmostEqual(y, y_ref[i], delta=XYZ_ATOL_M)
            self.assertAlmostEqual(z, z_ref[i], delta=XYZ_ATOL_M)


class CartesianToGeodeticVsPyprojTest(unittest.TestCase):
    """cartesian_to_geodetic must agree with pyproj to (near) floating point precision."""

    def test_random_points_global(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED))
        lat, lon, alt = _sample_geodetic_points(N_SAMPLES, rng)
        # Derive ECEF points from the reference transformer so the inverse
        # problem is well posed everywhere, including at the poles.
        x, y, z = _REFERENCE_TRANSFORMER.transform(lon, lat, alt, radians=False)

        lon_ref, lat_ref, alt_ref = _REFERENCE_TRANSFORMER.transform(
            x, y, z, radians=False, direction="INVERSE"
        )

        for i in range(N_SAMPLES):
            lat_c, lon_c, alt_c = CoordinateTransformations.cartesian_to_geodetic(
                float(x[i]), float(y[i]), float(z[i])
            )
            self.assertAlmostEqual(lat_c, lat_ref[i], delta=LAT_LON_ATOL_DEG)
            self.assertAlmostEqual(lon_c, lon_ref[i], delta=LAT_LON_ATOL_DEG)
            self.assertAlmostEqual(alt_c, alt_ref[i], delta=ALT_ATOL_M)

    def test_return_type_is_python_float(self) -> None:
        # ECEF coordinates of Paris (lat=48.8566, lon=2.3522, alt=35.0), taken
        # from the Octave geodetic2ecef reference in
        # test_coordinate_tansformations.py. Any valid ECEF point works here;
        # this one was picked simply because it's already a known-good
        # reference value elsewhere in the test suite.
        lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(
            4200937.804, 172560.7214, 4780107.699
        )
        self.assertIsInstance(lat, float)
        self.assertIsInstance(lon, float)
        self.assertIsInstance(alt, float)

    def test_poles(self) -> None:
        # WGS84 semi-minor axis [m]: at lat=+/-90 the ellipsoid normal is
        # exactly the z-axis, so a point at x=y=0, z=+/-(b + h) is *exactly*
        # geodetic (lat=+/-90, alt=h) -- no iterative/approximate solve needed
        # to construct this test point.
        b = 6356752.314245
        # h=500 is an arbitrary nonzero altitude: alt=0 would sit exactly on
        # the ellipsoid surface, which wouldn't exercise the altitude
        # computation at the pole singularity (x=y=0, where atan2-based
        # longitude/latitude formulas are most likely to misbehave).
        #
        # Note this is a different point than GeodeticToCartesianVsPyprojTest
        # .test_poles: that test fixes alt=0 and varies lon (checking the
        # forward transform ignores lon at the pole); this test fixes lon
        # implicitly via x=y=0 and uses alt=500 (checking the inverse
        # transform recovers a nonzero altitude at the pole). They are not a
        # forward/inverse round trip of the same point.
        h = 500.0
        for sign in (1.0, -1.0):
            x, y, z = 0.0, 0.0, sign * (b + h)
            lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
            lon_ref, lat_ref, alt_ref = _REFERENCE_TRANSFORMER.transform(
                x, y, z, radians=False, direction="INVERSE"
            )
            self.assertAlmostEqual(lat, lat_ref, delta=LAT_LON_ATOL_DEG)
            self.assertAlmostEqual(alt, alt_ref, delta=ALT_ATOL_M)

    def test_negative_and_high_altitude(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 1))
        lat, lon, alt = _sample_geodetic_points(
            500, rng, alt_range=(-10_000, 1_000_000)
        )
        x, y, z = _REFERENCE_TRANSFORMER.transform(lon, lat, alt, radians=False)
        lon_ref, lat_ref, alt_ref = _REFERENCE_TRANSFORMER.transform(
            x, y, z, radians=False, direction="INVERSE"
        )
        # See HIGH_ALTITUDE_*_ATOL_* above: pyproj's own inverse transform
        # loses precision at these altitudes, so this comparison needs looser
        # tolerances than the rest of the file.
        for i in range(500):
            lat_c, lon_c, alt_c = CoordinateTransformations.cartesian_to_geodetic(
                float(x[i]), float(y[i]), float(z[i])
            )
            self.assertAlmostEqual(
                lat_c, lat_ref[i], delta=HIGH_ALTITUDE_LAT_LON_ATOL_DEG
            )
            self.assertAlmostEqual(
                lon_c, lon_ref[i], delta=HIGH_ALTITUDE_LAT_LON_ATOL_DEG
            )
            self.assertAlmostEqual(alt_c, alt_ref[i], delta=HIGH_ALTITUDE_ALT_ATOL_M)


class PoleRoundTripTest(unittest.TestCase):
    """
    geodetic_to_cartesian -> cartesian_to_geodetic round trip, specifically at the poles.

    The pole-only tests above check each direction against pyproj independently,
    using different points (see the comment in
    CartesianToGeodeticVsPyprojTest.test_poles). This test instead pushes the
    same point through both directions of CoordinateTransformations and checks
    it survives, cross-checking against pyproj's own round trip of that point.
    """

    def test_round_trip_at_poles(self) -> None:
        for lat in (90.0, -90.0):
            for lon in (0.0, 45.0, 179.9, -179.9):
                for alt in (0.0, 500.0, -50.0):
                    x, y, z = CoordinateTransformations.geodetic_to_cartesian(
                        lat, lon, alt
                    )
                    lat_rt, _, alt_rt = CoordinateTransformations.cartesian_to_geodetic(
                        x, y, z
                    )

                    _, lat_pyproj_rt, alt_pyproj_rt = _REFERENCE_TRANSFORMER.transform(
                        x, y, z, radians=False, direction="INVERSE"
                    )

                    # Longitude is undefined at the poles (x=y=0), so we
                    # deliberately do not assert on lon_rt -- any value is
                    # geometrically equivalent there. We only check that
                    # lat/alt survive the round trip and agree with pyproj's
                    # own inverse of the same point.
                    self.assertAlmostEqual(lat_rt, lat, delta=LAT_LON_ATOL_DEG)
                    self.assertAlmostEqual(alt_rt, alt, delta=ALT_ATOL_M)
                    self.assertAlmostEqual(
                        lat_rt, lat_pyproj_rt, delta=LAT_LON_ATOL_DEG
                    )
                    self.assertAlmostEqual(alt_rt, alt_pyproj_rt, delta=ALT_ATOL_M)


if __name__ == "__main__":
    unittest.main()
