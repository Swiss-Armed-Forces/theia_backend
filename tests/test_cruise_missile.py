import math
import unittest

import numpy as np
from scipy.interpolate import CubicSpline

from theia.config import (
    CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE,
    CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE,
    CM_DEFAULT_TERMINAL_DIVE_ANGLE,
    CM_IMPACT_HOLD_OFFSETS,
    CM_PRE_IMPACT_OFFSETS,
)
from theia.cruise_missile import (
    build_terrain_following_path,
    default_min_clearance,
    great_circle_points,
    rate_limited_envelope,
)
from theia.distance import haversine
from theia.terrain import AbstractTerrainModel
from theia.types import Point

P_START = Point(lat=47.0, lon=8.0, alt=12345.0)
P_STOP = Point(lat=47.0, lon=8.4, alt=-500.0)
DISTANCE = haversine(P_START.lon, P_START.lat, P_STOP.lon, P_STOP.lat)
SPEED = 250.0
CRUISE_MAGL = 50.0


class FlatTerrain(AbstractTerrainModel):
    step_m: float = 30.0
    height: float = 400.0

    def elevationAt(self, lat, lon):
        return self.height


class RidgeTerrain(AbstractTerrainModel):
    """Gaussian ridge running north-south across the track."""

    step_m: float = 30.0
    base: float = 400.0
    height: float = 600.0
    lon_center: float = 8.2
    lon_width: float = 0.02

    def elevationAt(self, lat, lon):
        return self.base + self.height * math.exp(
            -(((lon - self.lon_center) / self.lon_width) ** 2)
        )


class ValleyTerrain(AbstractTerrainModel):
    """Plateau with a narrow, deep valley across the track."""

    step_m: float = 30.0

    def elevationAt(self, lat, lon):
        return 400.0 if abs(lon - 8.2) < 0.003 else 1000.0


DIVE_LENGTH = CRUISE_MAGL / math.tan(math.radians(-CM_DEFAULT_TERMINAL_DIVE_ANGLE))
"""Horizontal length [m] of the default terminal dive onto flat terrain."""


def distances_to_target(path) -> np.ndarray:
    """Ground distances [m] of the (non-hold) samples to the target."""
    n = len(path.times) - len(CM_IMPACT_HOLD_OFFSETS)
    return np.array(
        [
            haversine(lon, lat, P_STOP.lon, P_STOP.lat)
            for lat, lon in zip(path.lats[:n], path.lons[:n])
        ]
    )


def is_cruise(path) -> np.ndarray:
    """
    Whether each (non-hold) sample lies before or at the dive start.

    Assumes flat terrain around the target, so the dive has DIVE_LENGTH.
    """
    return distances_to_target(path) >= DIVE_LENGTH - 1e-3


def flight_path_angles(path, cruise_only: bool = False) -> np.ndarray:
    """Flight-path angles [°] between consecutive (non-hold) samples."""
    n = len(path.times) - len(CM_IMPACT_HOLD_OFFSETS)
    s = DISTANCE - distances_to_target(path)
    angles = np.degrees(np.arctan2(np.diff(path.alts[:n]), np.diff(s)))
    if cruise_only:
        # Segments ending at or before the dive start.
        angles = angles[is_cruise(path)[1:]]
    return angles


def build(terrain, **kwargs):
    return build_terrain_following_path(
        p_start=kwargs.pop("p_start", P_START),
        p_stop=kwargs.pop("p_stop", P_STOP),
        speed=kwargs.pop("speed", SPEED),
        cruise_magl=kwargs.pop("cruise_magl", CRUISE_MAGL),
        terrain=terrain,
        **kwargs,
    )


class GreatCircleTest(unittest.TestCase):
    def test_end_points(self):
        lats, lons = great_circle_points(P_START, P_STOP, np.array([0.0, DISTANCE]))
        np.testing.assert_allclose(lats, [P_START.lat, P_STOP.lat], atol=1e-9)
        np.testing.assert_allclose(lons, [P_START.lon, P_STOP.lon], atol=1e-9)

    def test_points_lie_on_great_circle(self):
        s = np.linspace(0, DISTANCE, 11)
        lats, lons = great_circle_points(P_START, P_STOP, s)
        for si, lat, lon in zip(s, lats, lons):
            d_start = haversine(P_START.lon, P_START.lat, lon, lat)
            d_stop = haversine(lon, lat, P_STOP.lon, P_STOP.lat)
            self.assertAlmostEqual(d_start, si, delta=1e-3)
            self.assertAlmostEqual(d_start + d_stop, DISTANCE, delta=1e-3)

    def test_great_circle_bends_north_of_parallel(self):
        # On the northern hemisphere the great circle between two points on the
        # same parallel lies north of that parallel.
        lats, _ = great_circle_points(P_START, P_STOP, np.array([DISTANCE / 2]))
        self.assertGreater(lats[0], P_START.lat)


class RateLimitedEnvelopeTest(unittest.TestCase):
    def test_flat_profile_unchanged(self):
        s = np.linspace(0, 1000, 11)
        z = rate_limited_envelope(s, np.full(11, 100.0), -0.1, 0.2)
        np.testing.assert_allclose(z, 100.0)

    def test_step_up_climbs_early(self):
        s = np.arange(0, 1001, 100.0)
        desired = np.where(s >= 500, 200.0, 100.0)
        z = rate_limited_envelope(s, desired, -0.1, 0.5)
        # 100 m climb at slope 0.5 starts 200 m before the step.
        np.testing.assert_allclose(z[:5], [100, 100, 100, 100, 150])
        np.testing.assert_allclose(z[5:], 200)

    def test_step_down_descends_slowly(self):
        s = np.arange(0, 1001, 100.0)
        desired = np.where(s >= 500, 100.0, 200.0)
        z = rate_limited_envelope(s, desired, -0.25, 0.5)
        np.testing.assert_allclose(z[:4], 200)
        np.testing.assert_allclose(z[4:], [200, 175, 150, 125, 100, 100, 100])

    def test_slopes_and_lower_bound(self):
        rng = np.random.default_rng(0)
        s = np.cumsum(rng.uniform(1, 50, 500))
        desired = rng.uniform(0, 300, 500)
        z = rate_limited_envelope(s, desired, -0.2, 0.3)
        slopes = np.diff(z) / np.diff(s)
        self.assertTrue((z >= desired).all())
        self.assertTrue((slopes <= 0.3 + 1e-9).all())
        self.assertTrue((slopes >= -0.2 - 1e-9).all())


class FlatTerrainTest(unittest.TestCase):
    def setUp(self):
        self.terrain = FlatTerrain()
        self.path = build(self.terrain)

    def test_start_altitude_ignores_given_altitude(self):
        self.assertAlmostEqual(self.path.alts[0], 400.0 + CRUISE_MAGL)

    def test_starts_and_ends_at_given_points(self):
        self.assertAlmostEqual(self.path.lats[0], P_START.lat)
        self.assertAlmostEqual(self.path.lons[0], P_START.lon)
        self.assertAlmostEqual(self.path.impact.lat, P_STOP.lat)
        self.assertAlmostEqual(self.path.impact.lon, P_STOP.lon)

    def test_impact_on_ground(self):
        self.assertAlmostEqual(self.path.impact.alt, 400.0)

    def test_constant_agl_until_dive(self):
        dive_length = CRUISE_MAGL / math.tan(
            math.radians(-CM_DEFAULT_TERMINAL_DIVE_ANGLE)
        )
        for lat, lon, alt in zip(self.path.lats, self.path.lons, self.path.alts):
            if haversine(lon, lat, P_STOP.lon, P_STOP.lat) > dive_length + 1:
                self.assertAlmostEqual(alt, 400.0 + CRUISE_MAGL)

    def test_terminal_dive_angle(self):
        angles = flight_path_angles(self.path)
        self.assertAlmostEqual(angles[-1], CM_DEFAULT_TERMINAL_DIVE_ANGLE, delta=0.01)

    def test_flight_time_matches_speed(self):
        dive_length = CRUISE_MAGL / math.tan(
            math.radians(-CM_DEFAULT_TERMINAL_DIVE_ANGLE)
        )
        path_length = DISTANCE - dive_length + math.hypot(dive_length, CRUISE_MAGL)
        self.assertAlmostEqual(self.path.t_impact, path_length / SPEED, delta=1e-3)

    def test_times_monotonic_and_relative_to_launch(self):
        self.assertEqual(self.path.times[0], 0.0)
        self.assertTrue((np.diff(self.path.times) > 0).all())

    def test_impact_is_held(self):
        n = len(CM_IMPACT_HOLD_OFFSETS)
        np.testing.assert_allclose(
            self.path.times[-n:], self.path.t_impact + np.array(CM_IMPACT_HOLD_OFFSETS)
        )
        for values in (self.path.lats, self.path.lons, self.path.alts):
            np.testing.assert_allclose(values[-n - 1 :], values[-1])

    def test_sample_spacing(self):
        path = build(self.terrain, sample_spacing=500.0)
        self.assertLess(len(path.times), len(self.path.times))
        n = len(path.times) - len(CM_IMPACT_HOLD_OFFSETS)
        steps = [
            haversine(lon1, lat1, lon2, lat2)
            for lat1, lon1, lat2, lon2 in zip(
                path.lats[: n - 1], path.lons[: n - 1], path.lats[1:n], path.lons[1:n]
            )
        ]
        self.assertLessEqual(max(steps), 500.0 + 1e-6)

    def test_hold_stays_near_impact(self):
        # The simulator must still see the missile within its combat range
        # (and above ground within interpolation noise) during the hold.
        spline = CubicSpline(
            self.path.times,
            np.stack([self.path.lats, self.path.lons, self.path.alts], axis=1),
        )
        t = np.linspace(self.path.t_impact, self.path.times[-1], 101)
        for lat, lon, alt in spline(t):
            self.assertLess(haversine(lon, lat, P_STOP.lon, P_STOP.lat), 30.0)
            self.assertGreater(alt, 400.0 - 5.0)


class RidgeTest(unittest.TestCase):
    def setUp(self):
        self.terrain = RidgeTerrain()
        self.path = build(self.terrain)
        n = len(self.path.times) - len(CM_IMPACT_HOLD_OFFSETS)
        self.agl = np.array(
            [
                alt - self.terrain.elevationAt(lat, lon)
                for lat, lon, alt in zip(
                    self.path.lats[:n], self.path.lons[:n], self.path.alts[:n]
                )
            ]
        )

    def test_never_below_cruise_magl_before_dive(self):
        self.assertTrue((self.agl[is_cruise(self.path)] >= CRUISE_MAGL - 1e-6).all())

    def test_flies_over_ridge(self):
        self.assertGreater(self.path.alts.max(), 400.0 + 600.0 + CRUISE_MAGL - 1)

    def test_climbs_before_steep_ridge(self):
        # Ahead of a ridge steeper than the climb limit, the missile is already
        # well above cruise_magl: it started climbing early.
        terrain = RidgeTerrain(lon_width=0.005)
        path = build(terrain)
        before_ridge = path.lons < terrain.lon_center
        excess = [
            alt - terrain.elevationAt(lat, lon) - CRUISE_MAGL
            for lat, lon, alt in zip(
                path.lats[before_ridge],
                path.lons[before_ridge],
                path.alts[before_ridge],
            )
        ]
        self.assertGreater(max(excess), 100.0)

    def test_flight_path_angles_within_limits(self):
        angles = flight_path_angles(self.path, cruise_only=True)
        self.assertLessEqual(angles.max(), CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE + 1e-3)
        self.assertGreaterEqual(angles.min(), CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE - 1e-3)
        # The limits are actually reached.
        self.assertGreater(angles.max(), CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE - 0.5)
        self.assertLess(angles.min(), CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE + 0.5)

    def test_asymmetric_limits(self):
        path = build(
            RidgeTerrain(lon_width=0.005),
            min_flight_path_angle=-5.0,
            max_flight_path_angle=25.0,
        )
        angles = flight_path_angles(path, cruise_only=True)
        self.assertLessEqual(angles.max(), 25.0 + 1e-3)
        self.assertGreaterEqual(angles.min(), -5.0 - 1e-3)
        self.assertGreater(angles.max(), 24.5)
        self.assertLess(angles.min(), -4.5)

    def test_interpolated_trajectory_keeps_min_clearance(self):
        spline = CubicSpline(
            self.path.times,
            np.stack([self.path.lats, self.path.lons, self.path.alts], axis=1),
        )
        t_dive = self.path.t_impact - math.hypot(DIVE_LENGTH, CRUISE_MAGL) / SPEED
        for lat, lon, alt in spline(np.linspace(0, t_dive, 2000)):
            self.assertGreaterEqual(
                alt - self.terrain.elevationAt(lat, lon),
                default_min_clearance(CRUISE_MAGL),
            )


class ValleyTest(unittest.TestCase):
    def test_stays_high_over_narrow_valley(self):
        path = build(ValleyTerrain())
        n = len(path.times) - len(CM_IMPACT_HOLD_OFFSETS)
        valley = np.abs(path.lons[:n] - 8.2) < 0.003
        self.assertTrue(valley.any())
        # The valley is ~450 m wide; descending at 10° only loses ~40 m per side.
        self.assertTrue((path.alts[:n][valley] > 1000.0).all())


class TerminalDiveTest(unittest.TestCase):
    def test_pre_impact_samples_on_dive_line(self):
        path = build(FlatTerrain())
        n = len(path.times) - len(CM_IMPACT_HOLD_OFFSETS)
        expected = path.t_impact - np.array(CM_PRE_IMPACT_OFFSETS)
        # Only offsets that fit into the dive (it lasts ~0.4 s here).
        dive_duration = math.hypot(DIVE_LENGTH, CRUISE_MAGL) / SPEED
        expected = expected[path.t_impact - expected < dive_duration]
        self.assertGreater(len(expected), 0)
        for t in expected:
            self.assertTrue(np.isclose(path.times[:n], t, atol=1e-6).any(), t)
        # All dive segments have the dive angle.
        dive_angles = flight_path_angles(path)[~is_cruise(path)[1:]]
        np.testing.assert_allclose(
            dive_angles, CM_DEFAULT_TERMINAL_DIVE_ANGLE, atol=0.01
        )

    def test_interpolated_position_near_impact(self):
        # The impact hold must not distort the approach: the interpolated
        # position stays close to the straight segments between the samples.
        path = build(FlatTerrain())
        n = len(path.times) - len(CM_IMPACT_HOLD_OFFSETS)
        spline = CubicSpline(
            path.times, np.stack([path.lats, path.lons, path.alts], axis=1)
        )
        t = np.linspace(path.t_impact - 1.0, path.t_impact, 500)
        lat, lon, alt = spline(t).T
        lat_lin = np.interp(t, path.times[:n], path.lats[:n])
        lon_lin = np.interp(t, path.times[:n], path.lons[:n])
        alt_lin = np.interp(t, path.times[:n], path.alts[:n])
        horizontal = np.array(
            [haversine(a, b, c, d) for a, b, c, d in zip(lon, lat, lon_lin, lat_lin)]
        )
        self.assertLess(np.hypot(horizontal, alt - alt_lin).max(), 5.0)

    def test_custom_dive_angle(self):
        path = build(FlatTerrain(), terminal_dive_angle=-60.0)
        self.assertAlmostEqual(flight_path_angles(path)[-1], -60.0, delta=0.01)

    def test_dive_onto_elevated_target(self):
        # The target sits on a hill top: impact at the hill's elevation.
        terrain = RidgeTerrain(lon_center=8.4, lon_width=0.05)
        path = build(terrain)
        self.assertAlmostEqual(path.impact.alt, 1000.0, delta=1e-6)

    def test_dive_blocked(self):
        # A 200 m ridge 300-600 m before the target sticks out of a -5° dive.
        class RidgeBeforeTarget(AbstractTerrainModel):
            step_m: float = 30.0

            def elevationAt(self, lat, lon):
                return (
                    600.0 if P_STOP.lon - 0.0079 < lon < P_STOP.lon - 0.004 else 400.0
                )

        with self.assertRaisesRegex(ValueError, "blocked"):
            build(RidgeBeforeTarget(), terminal_dive_angle=-5.0)

    def test_steep_dive_clears_ridge_before_target(self):
        class RidgeBeforeTarget(AbstractTerrainModel):
            step_m: float = 30.0

            def elevationAt(self, lat, lon):
                return (
                    450.0 if P_STOP.lon - 0.0079 < lon < P_STOP.lon - 0.004 else 400.0
                )

        path = build(RidgeBeforeTarget(), terminal_dive_angle=-60.0)
        self.assertAlmostEqual(path.impact.alt, 400.0)


class ValidationTest(unittest.TestCase):
    def assert_invalid(self, pattern, **kwargs):
        with self.assertRaisesRegex(ValueError, pattern):
            build(FlatTerrain(), **kwargs)

    def test_invalid_parameters(self):
        self.assert_invalid("speed", speed=0.0)
        self.assert_invalid("cruise_magl", cruise_magl=-1.0)
        self.assert_invalid("min_flight_path_angle", min_flight_path_angle=0.0)
        self.assert_invalid("min_flight_path_angle", min_flight_path_angle=-90.0)
        self.assert_invalid("max_flight_path_angle", max_flight_path_angle=0.0)
        self.assert_invalid("max_flight_path_angle", max_flight_path_angle=90.0)
        self.assert_invalid("terminal_dive_angle", terminal_dive_angle=0.0)
        self.assert_invalid("terminal_dive_angle", terminal_dive_angle=-90.0)
        # Positive values (the former convention) are rejected.
        self.assert_invalid("terminal_dive_angle", terminal_dive_angle=30.0)
        self.assert_invalid("min_clearance", min_clearance=0.0)
        self.assert_invalid("min_clearance", min_clearance=CRUISE_MAGL + 1)
        self.assert_invalid("sample_spacing", sample_spacing=0.0)

    def test_start_equals_target(self):
        self.assert_invalid("apart", p_stop=P_START)

    def test_target_too_close_for_dive(self):
        # 300 m range, but a -5° dive from 50 m AGL needs ~570 m.
        p_stop = Point(lat=47.0, lon=8.004, alt=0.0)
        self.assert_invalid("too close", p_stop=p_stop, terminal_dive_angle=-5.0)

    def test_launched_higher_to_clear_terrain_near_start(self):
        # A 300 m wall 200 m after launch cannot be cleared at 15° from 50 m
        # AGL, so the missile is launched higher instead.
        class Wall(AbstractTerrainModel):
            step_m: float = 30.0

            def elevationAt(self, lat, lon):
                return 700.0 if lon > 8.0027 else 400.0

        path = build(Wall())
        n = len(path.times) - len(CM_IMPACT_HOLD_OFFSETS)
        agl = np.array(
            [
                alt - Wall().elevationAt(lat, lon)
                for lat, lon, alt in zip(path.lats[:n], path.lons[:n], path.alts[:n])
            ]
        )
        self.assertGreater(path.alts[0], 400.0 + CRUISE_MAGL + 200.0)
        self.assertTrue((agl[is_cruise(path)] >= CRUISE_MAGL - 1e-6).all())
        # Climbing at the limit right from the launch.
        angles = flight_path_angles(path, cruise_only=True)
        self.assertAlmostEqual(angles[0], CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE, delta=0.01)
        self.assertLessEqual(angles.max(), CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE + 1e-3)

    def test_default_min_clearance(self):
        self.assertEqual(default_min_clearance(100.0), 50.0)
        self.assertEqual(default_min_clearance(20.0), 15.0)
        # Never above the cruise height itself.
        self.assertEqual(default_min_clearance(10.0), 10.0)


if __name__ == "__main__":
    unittest.main()
