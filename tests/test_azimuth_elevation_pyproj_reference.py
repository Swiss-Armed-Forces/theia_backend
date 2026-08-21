"""
Reference tests for calculate_azimuth_angle and calculate_elevation_angle
against independent oracles.

Azimuth
-------
calculate_azimuth_angle is atan2(east, north) of the target's position in the
observer's ellipsoidal-normal ENU frame -- i.e. the straight-line ("chord")
bearing in the local tangent plane, the same quantity
EcefToEnuTransformer/PROJ's `+proj=topocentric` compute. So the oracle here
is the same one used in test_ecef_to_enu_pyproj_reference.py: build east/north
via `+proj=topocentric` independently, then atan2 them the same way.

NOT the geodesic forward azimuth pyproj.Geod.inv would give -- that follows
the ellipsoid's surface and diverges from the chord bearing at long range.
Testing against Geod.inv would be testing a different (if more common)
definition than the one this codebase actually implements.

Elevation
---------
calculate_elevation_angle_ecef uses "up" = the observer's geocentric radial
direction (normalized ECEF position vector), NOT the ellipsoidal-normal "up"
used by the ENU frame above. On the WGS84 ellipsoid these two "up" directions
coincide only at the equator and poles; elsewhere they differ by the angle
between geocentric and geodetic latitude (up to ~0.19 deg on WGS84). This is
a real inconsistency between how this module defines "up" for azimuth vs.
elevation, not a bug introduced here -- preserved as-is since this file only
freezes current behavior for a performance-refactor, not a correctness fix.
No standard library function computes elevation with a geocentric "up", so
the oracle below re-derives the same formula independently in plain numpy
(sourcing ECEF coordinates from pyproj's `+proj=cart`, not from
CoordinateTransformations), rather than reusing any pyproj angle operator.
"""

import unittest

import numpy as np
import pyproj

from theia.coordinates import calculate_azimuth_angle, calculate_elevation_angle
from theia.types import Point

_CART = pyproj.Transformer.from_pipeline("+proj=cart +ellps=WGS84")


def _topocentric_transformer(lat0: float, lon0: float, h0: float) -> pyproj.Transformer:
    return pyproj.Transformer.from_pipeline(
        f"+proj=topocentric +ellps=WGS84 +lat_0={lat0} +lon_0={lon0} +h_0={h0}"
    )


SEED = 20260821
N_SAMPLES = 1_000

# Both oracles reduce to closed-form vector math (no iterative solve), same
# as EcefToEnuTransformer, measured worst-case agreement between the
# current implementation and these oracles, over N_SAMPLES points spanning
# the whole globe plus poles and altitudes to 1_000_000 m, is ~3e-15 rad
# (float64 rounding). These tolerances carry a >1e5x margin over that.
ANGLE_ATOL_RAD = 1e-9


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


def _reference_azimuth(
    obs_lat: float, obs_lon: float, obs_alt: float, tx: float, ty: float, tz: float
) -> float:
    t = _topocentric_transformer(obs_lat, obs_lon, obs_alt)
    east_ref, north_ref, _ = t.transform(tx, ty, tz)
    return float(np.arctan2(east_ref, north_ref) % (2 * np.pi))


def _reference_elevation(
    ox: float, oy: float, oz: float, tx: float, ty: float, tz: float
) -> float:
    obs_ecef = np.array([ox, oy, oz])
    tgt_ecef = np.array([tx, ty, tz])
    delta = tgt_ecef - obs_ecef
    up = obs_ecef / np.linalg.norm(obs_ecef)
    return float(np.arcsin(np.dot(up, delta) / np.linalg.norm(delta)))


class CalculateAzimuthAngleVsPyprojTest(unittest.TestCase):
    def test_random_points_global(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED))
        obs_lat, obs_lon, obs_alt = _sample_geodetic_points(N_SAMPLES, rng)
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(N_SAMPLES, rng)
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)

        for i in range(N_SAMPLES):
            p_obs = Point(
                lat=float(obs_lat[i]), lon=float(obs_lon[i]), alt=float(obs_alt[i])
            )
            p_tgt = Point(
                lat=float(tgt_lat[i]), lon=float(tgt_lon[i]), alt=float(tgt_alt[i])
            )
            az = calculate_azimuth_angle(p_obs, p_tgt)
            az_ref = _reference_azimuth(
                obs_lat[i], obs_lon[i], obs_alt[i], tx[i], ty[i], tz[i]
            )
            self.assertAlmostEqual(az, az_ref, delta=ANGLE_ATOL_RAD)

    def test_return_type_is_python_float(self) -> None:
        p_obs = Point(lat=46.8, lon=8.2, alt=500.0)
        p_tgt = Point(lat=48.8566, lon=2.3522, alt=35.0)
        az = calculate_azimuth_angle(p_obs, p_tgt)
        self.assertIsInstance(az, float)

    def test_result_in_0_2pi(self) -> None:
        """azimuth is documented to be in (0, 2pi)."""
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 1))
        obs_lat, obs_lon, obs_alt = _sample_geodetic_points(500, rng)
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(500, rng)
        for i in range(500):
            p_obs = Point(
                lat=float(obs_lat[i]), lon=float(obs_lon[i]), alt=float(obs_alt[i])
            )
            p_tgt = Point(
                lat=float(tgt_lat[i]), lon=float(tgt_lon[i]), alt=float(tgt_alt[i])
            )
            az = calculate_azimuth_angle(p_obs, p_tgt)
            self.assertGreaterEqual(az, 0.0)
            self.assertLess(az, 2 * np.pi)

    def test_poles(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 2))
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(200, rng)
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)

        for obs_lat in (90.0, -90.0):
            for obs_lon in (0.0, 45.0, 179.9, -179.9):
                p_obs = Point(lat=obs_lat, lon=obs_lon, alt=250.0)
                for i in range(200):
                    p_tgt = Point(
                        lat=float(tgt_lat[i]),
                        lon=float(tgt_lon[i]),
                        alt=float(tgt_alt[i]),
                    )
                    az = calculate_azimuth_angle(p_obs, p_tgt)
                    az_ref = _reference_azimuth(
                        obs_lat, obs_lon, 250.0, tx[i], ty[i], tz[i]
                    )
                    self.assertAlmostEqual(az, az_ref, delta=ANGLE_ATOL_RAD)

    def test_negative_and_high_altitude(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 3))
        alt_range = (-10_000, 1_000_000)
        obs_lat, obs_lon, obs_alt = _sample_geodetic_points(
            300, rng, alt_range=alt_range
        )
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(
            300, rng, alt_range=alt_range
        )
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)

        for i in range(300):
            p_obs = Point(
                lat=float(obs_lat[i]), lon=float(obs_lon[i]), alt=float(obs_alt[i])
            )
            p_tgt = Point(
                lat=float(tgt_lat[i]), lon=float(tgt_lon[i]), alt=float(tgt_alt[i])
            )
            az = calculate_azimuth_angle(p_obs, p_tgt)
            az_ref = _reference_azimuth(
                obs_lat[i], obs_lon[i], obs_alt[i], tx[i], ty[i], tz[i]
            )
            self.assertAlmostEqual(az, az_ref, delta=ANGLE_ATOL_RAD)


class CalculateElevationAngleVsManualReferenceTest(unittest.TestCase):
    def test_random_points_global(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED))
        obs_lat, obs_lon, obs_alt = _sample_geodetic_points(N_SAMPLES, rng)
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(N_SAMPLES, rng)
        ox, oy, oz = _CART.transform(obs_lon, obs_lat, obs_alt)
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)

        for i in range(N_SAMPLES):
            p_obs = Point(
                lat=float(obs_lat[i]), lon=float(obs_lon[i]), alt=float(obs_alt[i])
            )
            p_tgt = Point(
                lat=float(tgt_lat[i]), lon=float(tgt_lon[i]), alt=float(tgt_alt[i])
            )
            el = calculate_elevation_angle(p_obs, p_tgt)
            el_ref = _reference_elevation(
                ox[i], oy[i], oz[i], tx[i], ty[i], tz[i]
            )
            self.assertAlmostEqual(el, el_ref, delta=ANGLE_ATOL_RAD)

    def test_return_type_is_python_float(self) -> None:
        p_obs = Point(lat=46.8, lon=8.2, alt=500.0)
        p_tgt = Point(lat=48.8566, lon=2.3522, alt=35.0)
        el = calculate_elevation_angle(p_obs, p_tgt)
        self.assertIsInstance(el, float)

    def test_result_in_minus_half_pi_half_pi(self) -> None:
        """elevation is documented to be in [-pi/2, pi/2]."""
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 1))
        obs_lat, obs_lon, obs_alt = _sample_geodetic_points(500, rng)
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(500, rng)
        for i in range(500):
            p_obs = Point(
                lat=float(obs_lat[i]), lon=float(obs_lon[i]), alt=float(obs_alt[i])
            )
            p_tgt = Point(
                lat=float(tgt_lat[i]), lon=float(tgt_lon[i]), alt=float(tgt_alt[i])
            )
            el = calculate_elevation_angle(p_obs, p_tgt)
            self.assertGreaterEqual(el, -np.pi / 2)
            self.assertLessEqual(el, np.pi / 2)

    def test_poles(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 2))
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(200, rng)
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)

        for obs_lat in (90.0, -90.0):
            for obs_lon in (0.0, 45.0, 179.9, -179.9):
                p_obs = Point(lat=obs_lat, lon=obs_lon, alt=250.0)
                ox, oy, oz = _CART.transform(obs_lon, obs_lat, 250.0)
                for i in range(200):
                    p_tgt = Point(
                        lat=float(tgt_lat[i]),
                        lon=float(tgt_lon[i]),
                        alt=float(tgt_alt[i]),
                    )
                    el = calculate_elevation_angle(p_obs, p_tgt)
                    el_ref = _reference_elevation(
                        ox, oy, oz, tx[i], ty[i], tz[i]
                    )
                    self.assertAlmostEqual(el, el_ref, delta=ANGLE_ATOL_RAD)

    def test_negative_and_high_altitude(self) -> None:
        rng = np.random.Generator(np.random.PCG64(seed=SEED + 3))
        alt_range = (-10_000, 1_000_000)
        obs_lat, obs_lon, obs_alt = _sample_geodetic_points(
            300, rng, alt_range=alt_range
        )
        tgt_lat, tgt_lon, tgt_alt = _sample_geodetic_points(
            300, rng, alt_range=alt_range
        )
        ox, oy, oz = _CART.transform(obs_lon, obs_lat, obs_alt)
        tx, ty, tz = _CART.transform(tgt_lon, tgt_lat, tgt_alt)

        for i in range(300):
            p_obs = Point(
                lat=float(obs_lat[i]), lon=float(obs_lon[i]), alt=float(obs_alt[i])
            )
            p_tgt = Point(
                lat=float(tgt_lat[i]), lon=float(tgt_lon[i]), alt=float(tgt_alt[i])
            )
            el = calculate_elevation_angle(p_obs, p_tgt)
            el_ref = _reference_elevation(
                ox[i], oy[i], oz[i], tx[i], ty[i], tz[i]
            )
            self.assertAlmostEqual(el, el_ref, delta=ANGLE_ATOL_RAD)


if __name__ == "__main__":
    unittest.main()
