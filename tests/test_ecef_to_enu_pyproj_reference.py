"""
Reference tests for EcefToEnuTransformer against an independent pyproj oracle.

PROJ's `+proj=topocentric` operator converts ECEF (X, Y, Z) directly to a local
East-North-Up frame given a geodetic origin (lat_0, lon_0, h_0) -- the same
operation EcefToEnuTransformer implements by hand (rotation matrix +
translation). Chained with `+proj=cart` (geodetic -> ECEF) to build target
points, this gives a full, independent (pyproj/PROJ, not theia) ground-truth
oracle for both ecef_to_enu and enu_to_ecef, the same role
tests/test_coordinate_transformations_pyproj_reference.py plays for
CoordinateTransformations.
"""

import unittest

import numpy as np
import pyproj

from theia.coordinates import EcefToEnuTransformer
from theia.types import Point

# Independent oracle transformers, deliberately not shared with the
# implementation under test. `_CART` does geodetic -> ECEF (used only to
# construct target/reference points in ECEF); `_topocentric_transformer`
# builds a fresh ECEF <-> ENU transformer for one particular origin.
_CART = pyproj.Transformer.from_pipeline("+proj=cart +ellps=WGS84")


def _topocentric_transformer(lat0: float, lon0: float, h0: float) -> pyproj.Transformer:
    return pyproj.Transformer.from_pipeline(
        f"+proj=topocentric +ellps=WGS84 +lat_0={lat0} +lon_0={lon0} +h_0={h0}"
    )


SEED = 20260821
N_SAMPLES = 2_000

# Absolute tolerance [m].
# Measured directly (comparing EcefToEnuTransformer's current implementation
# against this file's oracle over N_SAMPLES points spanning the whole globe,
# reference points at the poles, and altitudes from -10_000 m to 1_000_000 m)
# the worst-case disagreement is ~1.9e-9 m -- i.e. float64 rounding, not
# algorithmic error. ENU_ATOL_M carries a >100x margin over that.
ENU_ATOL_M = 1e-6


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


class EcefToEnuVsPyprojTest(unittest.TestCase):
    """EcefToEnuTransformer.ecef_to_enu must agree with PROJ's topocentric operator."""

    def test_random_points_global(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED))
        ref_lat, ref_lon, ref_alt = _sample_geodetic_points(N_SAMPLES, rng)
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(N_SAMPLES, rng)
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)

        for i in range(N_SAMPLES):
            reference = Point(
                lat=float(ref_lat[i]), lon=float(ref_lon[i]), alt=float(ref_alt[i])
            )
            transformer = EcefToEnuTransformer(reference)
            east, north, up = transformer.ecef_to_enu(
                (float(tx[i]), float(ty[i]), float(tz[i]))
            )

            t = _topocentric_transformer(ref_lat[i], ref_lon[i], ref_alt[i])
            east_ref, north_ref, up_ref = t.transform(tx[i], ty[i], tz[i])

            self.assertAlmostEqual(east, east_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(north, north_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(up, up_ref, delta=ENU_ATOL_M)

    def test_return_type_is_python_float(self) -> None:
        reference = Point(lat=46.8, lon=8.2, alt=500.0)
        transformer = EcefToEnuTransformer(reference)
        # Target point: ECEF coordinates of Paris (lat=48.8566, lon=2.3522,
        # alt=35.0), the same known-good reference point used in
        # test_coordinate_transformations_pyproj_reference.py. Any valid ECEF
        # point works here for a type check; reusing this one avoids
        # introducing an unexplained new triple.
        east, north, up = transformer.ecef_to_enu(
            (4200937.804, 172560.7214, 4780107.699)
        )
        self.assertIsInstance(east, float)
        self.assertIsInstance(north, float)
        self.assertIsInstance(up, float)

    def test_reference_point_is_origin(self) -> None:
        """The reference point itself must transform to ENU (0, 0, 0)."""
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 1))
        ref_lat, ref_lon, ref_alt = _sample_geodetic_points(50, rng)
        rx, ry, rz = _CART.transform(ref_lon, ref_lat, ref_alt)

        for i in range(50):
            reference = Point(
                lat=float(ref_lat[i]), lon=float(ref_lon[i]), alt=float(ref_alt[i])
            )
            transformer = EcefToEnuTransformer(reference)
            east, north, up = transformer.ecef_to_enu(
                (float(rx[i]), float(ry[i]), float(rz[i]))
            )

            # The actual claim in the docstring: ENU (0, 0, 0), checked
            # directly rather than only via agreement with pyproj (pyproj
            # agreeing with us on a nonzero value would not prove either of
            # us is right about the origin actually being zero).
            self.assertAlmostEqual(east, 0.0, delta=ENU_ATOL_M)
            self.assertAlmostEqual(north, 0.0, delta=ENU_ATOL_M)
            self.assertAlmostEqual(up, 0.0, delta=ENU_ATOL_M)

            # Cross-check against pyproj too, consistent with the rest of
            # this file.
            t = _topocentric_transformer(ref_lat[i], ref_lon[i], ref_alt[i])
            east_ref, north_ref, up_ref = t.transform(rx[i], ry[i], rz[i])

            self.assertAlmostEqual(east, east_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(north, north_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(up, up_ref, delta=ENU_ATOL_M)

    def test_poles(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 2))
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(200, rng)
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)

        for ref_lat in (90.0, -90.0):
            for ref_lon in (0.0, 45.0, 179.9, -179.9):
                reference = Point(lat=ref_lat, lon=ref_lon, alt=250.0)
                transformer = EcefToEnuTransformer(reference)
                t = _topocentric_transformer(ref_lat, ref_lon, 250.0)
                for i in range(200):
                    east, north, up = transformer.ecef_to_enu(
                        (float(tx[i]), float(ty[i]), float(tz[i]))
                    )
                    east_ref, north_ref, up_ref = t.transform(tx[i], ty[i], tz[i])
                    self.assertAlmostEqual(east, east_ref, delta=ENU_ATOL_M)
                    self.assertAlmostEqual(north, north_ref, delta=ENU_ATOL_M)
                    self.assertAlmostEqual(up, up_ref, delta=ENU_ATOL_M)

    def test_negative_and_high_altitude(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 3))
        alt_range = (-10_000, 1_000_000)
        ref_lat, ref_lon, ref_alt = _sample_geodetic_points(
            300, rng, alt_range=alt_range
        )
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(
            300, rng, alt_range=alt_range
        )
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)

        for i in range(300):
            reference = Point(
                lat=float(ref_lat[i]), lon=float(ref_lon[i]), alt=float(ref_alt[i])
            )
            transformer = EcefToEnuTransformer(reference)
            east, north, up = transformer.ecef_to_enu(
                (float(tx[i]), float(ty[i]), float(tz[i]))
            )

            t = _topocentric_transformer(ref_lat[i], ref_lon[i], ref_alt[i])
            east_ref, north_ref, up_ref = t.transform(tx[i], ty[i], tz[i])

            self.assertAlmostEqual(east, east_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(north, north_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(up, up_ref, delta=ENU_ATOL_M)


class EnuToEcefVsPyprojTest(unittest.TestCase):
    """EcefToEnuTransformer.enu_to_ecef must agree with PROJ's topocentric operator (inverse)."""

    def test_random_points_global(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED))
        ref_lat, ref_lon, ref_alt = _sample_geodetic_points(N_SAMPLES, rng)
        east = rng.uniform(-1_000_000, 1_000_000, size=N_SAMPLES)
        north = rng.uniform(-1_000_000, 1_000_000, size=N_SAMPLES)
        up = rng.uniform(-1_000_000, 1_000_000, size=N_SAMPLES)

        for i in range(N_SAMPLES):
            reference = Point(
                lat=float(ref_lat[i]), lon=float(ref_lon[i]), alt=float(ref_alt[i])
            )
            transformer = EcefToEnuTransformer(reference)
            x, y, z = transformer.enu_to_ecef(
                (float(east[i]), float(north[i]), float(up[i]))
            )

            t = _topocentric_transformer(ref_lat[i], ref_lon[i], ref_alt[i])
            x_ref, y_ref, z_ref = t.transform(
                east[i], north[i], up[i], direction="INVERSE"
            )

            self.assertAlmostEqual(x, x_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(y, y_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(z, z_ref, delta=ENU_ATOL_M)

    def test_return_type_is_python_float(self) -> None:
        reference = Point(lat=46.8, lon=8.2, alt=500.0)
        transformer = EcefToEnuTransformer(reference)
        x, y, z = transformer.enu_to_ecef((100.0, 200.0, 300.0))
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertIsInstance(z, float)

    def test_poles(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 2))
        east = rng.uniform(-1_000_000, 1_000_000, size=200)
        north = rng.uniform(-1_000_000, 1_000_000, size=200)
        up = rng.uniform(-1_000_000, 1_000_000, size=200)

        for ref_lat in (90.0, -90.0):
            for ref_lon in (0.0, 45.0, 179.9, -179.9):
                reference = Point(lat=ref_lat, lon=ref_lon, alt=250.0)
                transformer = EcefToEnuTransformer(reference)
                t = _topocentric_transformer(ref_lat, ref_lon, 250.0)
                for i in range(200):
                    x, y, z = transformer.enu_to_ecef(
                        (float(east[i]), float(north[i]), float(up[i]))
                    )
                    x_ref, y_ref, z_ref = t.transform(
                        east[i], north[i], up[i], direction="INVERSE"
                    )
                    self.assertAlmostEqual(x, x_ref, delta=ENU_ATOL_M)
                    self.assertAlmostEqual(y, y_ref, delta=ENU_ATOL_M)
                    self.assertAlmostEqual(z, z_ref, delta=ENU_ATOL_M)

    def test_negative_and_high_altitude(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 3))
        ref_lat, ref_lon, ref_alt = _sample_geodetic_points(
            300, rng, alt_range=(-10_000, 1_000_000)
        )
        east = rng.uniform(-1_000_000, 1_000_000, size=300)
        north = rng.uniform(-1_000_000, 1_000_000, size=300)
        up = rng.uniform(-1_000_000, 1_000_000, size=300)

        for i in range(300):
            reference = Point(
                lat=float(ref_lat[i]), lon=float(ref_lon[i]), alt=float(ref_alt[i])
            )
            transformer = EcefToEnuTransformer(reference)
            x, y, z = transformer.enu_to_ecef(
                (float(east[i]), float(north[i]), float(up[i]))
            )

            t = _topocentric_transformer(ref_lat[i], ref_lon[i], ref_alt[i])
            x_ref, y_ref, z_ref = t.transform(
                east[i], north[i], up[i], direction="INVERSE"
            )

            self.assertAlmostEqual(x, x_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(y, y_ref, delta=ENU_ATOL_M)
            self.assertAlmostEqual(z, z_ref, delta=ENU_ATOL_M)


class EcefToEnuMultipleVsPyprojTest(unittest.TestCase):
    """ecef_to_enu_multiple (the batched/vectorized path) must agree with the oracle too."""

    def test_matches_pyproj_batch(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 4))
        reference = Point(lat=47.0, lon=8.0, alt=1200.0)
        transformer = EcefToEnuTransformer(reference)

        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(500, rng)
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)
        points_ecef = list(zip(tx.tolist(), ty.tolist(), tz.tolist()))

        result = transformer.ecef_to_enu_multiple(points_ecef)

        t = _topocentric_transformer(reference.lat, reference.lon, reference.alt)
        east_ref, north_ref, up_ref = t.transform(tx, ty, tz)
        expected = np.stack([east_ref, north_ref, up_ref], axis=1)

        self.assertTrue(np.allclose(result, expected, atol=ENU_ATOL_M))


if __name__ == "__main__":
    unittest.main()
