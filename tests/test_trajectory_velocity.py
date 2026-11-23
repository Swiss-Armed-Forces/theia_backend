import datetime
import unittest

import numpy as np
from scipy.interpolate import CubicSpline

from theia.coordinates import CoordinateTransformations
from theia.types import (
    ConstantRcsModel,
    Point,
    TargetInfos,
    Trajectory,
    _CubicSplineEvaluator,
)

T0 = datetime.datetime(2026, 11, 20, tzinfo=datetime.UTC)


def _trajectory(
    lats: list[float],
    lons: list[float],
    alts: list[float],
    dt_s: float = 10.0,
) -> Trajectory:
    return Trajectory(
        target_id=0,
        target_info=TargetInfos.FIXED_WING,
        times=[T0 + datetime.timedelta(seconds=dt_s * i) for i in range(len(lats))],
        lats=lats,
        lons=lons,
        alts=alts,
        cross_section_model=ConstantRcsModel(rcs=1.0),
    )


def _ecef(trajectory: Trajectory, t: datetime.datetime) -> np.ndarray:
    target = trajectory(t)
    return np.array(
        CoordinateTransformations.geodetic_to_cartesian(*target.point.as_tuple())
    )


class GeodeticRateToEcefVelocityTest(unittest.TestCase):
    def test_matches_finite_difference(self):
        # Central differences of geodetic_to_cartesian along a straight move in
        # geodetic coordinates; the error is O(h^2), far below the tolerance.
        h = 1e-3
        cases = [
            # (lat, lon, alt, lat_rate [°/s], lon_rate [°/s], alt_rate [m/s])
            (47.4, 8.5, 500.0, 1e-3, 2e-3, 10.0),
            (-30.0, 120.0, 10_000.0, -5e-4, 1e-3, -50.0),
            (80.0, -60.0, 0.0, 2e-4, 5e-3, 0.0),
            (0.0, 179.0, 30_000.0, 0.0, -1e-3, 100.0),
            (47.0, 8.0, 1_000.0, 0.0, 0.0, 0.0),
        ]
        for lat, lon, alt, dlat, dlon, dalt in cases:
            with self.subTest(lat=lat, lon=lon, alt=alt):
                before = CoordinateTransformations.geodetic_to_cartesian(
                    lat - dlat * h, lon - dlon * h, alt - dalt * h
                )
                after = CoordinateTransformations.geodetic_to_cartesian(
                    lat + dlat * h, lon + dlon * h, alt + dalt * h
                )
                expected = (np.array(after) - np.array(before)) / (2 * h)

                v = CoordinateTransformations.geodetic_rate_to_ecef_velocity(
                    lat, lon, alt, dlat, dlon, dalt
                )

                np.testing.assert_allclose(v.as_tuple(), expected, atol=1e-4)

    def test_pure_climb_points_up(self):
        lat, lon = 47.0, 8.0
        v = CoordinateTransformations.geodetic_rate_to_ecef_velocity(
            lat, lon, 500.0, 0.0, 0.0, 10.0
        )
        v_north, v_east, v_up = CoordinateTransformations.velocity_ecef_to_neu(
            Point(lat=lat, lon=lon, alt=500.0), v
        )
        np.testing.assert_allclose((v_north, v_east, v_up), (0.0, 0.0, 10.0), atol=1e-9)


class CubicSplineEvaluatorDerivativeTest(unittest.TestCase):
    def test_matches_scipy_derivative(self):
        rng = np.random.default_rng(20261120)
        x = np.cumsum(rng.uniform(1.0, 10.0, size=8))
        y = rng.normal(size=(8, 3))
        spline = CubicSpline(x, y, extrapolate=True)
        evaluator = _CubicSplineEvaluator(spline)
        derivative = spline.derivative()

        # Includes points outside [x[0], x[-1]], which are extrapolated.
        for xi in np.linspace(x[0] - 2.0, x[-1] + 2.0, 50):
            values, derivatives = evaluator.with_derivative(float(xi))
            np.testing.assert_allclose(values, spline(xi), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                derivatives, derivative(xi), rtol=1e-10, atol=1e-12
            )


class TrajectoryVelocityTest(unittest.TestCase):
    def test_velocity_matches_change_of_position(self):
        trajectory = _trajectory(
            lats=[47.0, 47.1, 47.15, 47.3, 47.32],
            lons=[8.0, 8.05, 8.2, 8.21, 8.4],
            alts=[100.0, 500.0, 450.0, 1_000.0, 300.0],
        )
        # h trades round-off (POSIX timestamps ~1.8e9 s, geodetic round
        # trips) against truncation error; 10 ms keeps both at ~mm/s.
        h = datetime.timedelta(milliseconds=10)
        for seconds in [0.5, 7.0, 13.3, 25.0, 39.5]:
            with self.subTest(seconds=seconds):
                t = T0 + datetime.timedelta(seconds=seconds)
                expected = (_ecef(trajectory, t + h) - _ecef(trajectory, t - h)) / (
                    2 * h.total_seconds()
                )

                v = trajectory(t).velocity

                # Speeds are ~1-3 km/s here.
                np.testing.assert_allclose(v.as_tuple(), expected, atol=1e-2)

    def test_velocity_at_waypoints_is_finite(self):
        trajectory = _trajectory(
            lats=[47.0, 47.01, 47.02],
            lons=[8.0, 8.0, 8.0],
            alts=[500.0, 500.0, 500.0],
        )
        for t in trajectory.times:
            with self.subTest(t=t):
                self.assertTrue(np.isfinite(trajectory(t).velocity.as_tuple()).all())

    def test_constant_northward_flight(self):
        # 0.01° of latitude in 10 s at constant altitude.
        trajectory = _trajectory(
            lats=[47.0, 47.01, 47.02],
            lons=[8.0, 8.0, 8.0],
            alts=[500.0, 500.0, 500.0],
        )
        t = T0 + datetime.timedelta(seconds=10)
        target = trajectory(t)

        v_north, v_east, v_up = CoordinateTransformations.velocity_ecef_to_neu(
            target.point, target.velocity
        )

        # Meridian arc of 0.01° at 47° latitude ~= 1112 m.
        self.assertAlmostEqual(v_north, 111.2, delta=0.5)
        self.assertAlmostEqual(v_east, 0.0, delta=1e-6)
        self.assertAlmostEqual(v_up, 0.0, delta=1e-6)

    def test_stationary_trajectory_has_zero_velocity(self):
        trajectory = _trajectory(
            lats=[47.0, 47.0],
            lons=[8.0, 8.0],
            alts=[500.0, 500.0],
        )
        target = trajectory(T0 + datetime.timedelta(seconds=5))
        self.assertEqual(target.velocity.as_tuple(), (0.0, 0.0, 0.0))


class LegacyTrajectoryFileTest(unittest.TestCase):
    def test_stored_velocities_are_ignored(self):
        # Files written before velocities were derived contain vxs, vys, vzs.
        data = {
            "target_id": 3,
            "target_info": TargetInfos.FIXED_WING.model_dump(mode="json"),
            "times": [
                T0.isoformat(),
                (T0 + datetime.timedelta(seconds=1)).isoformat(),
            ],
            "lats": [47.0, 47.0],
            "lons": [8.0, 8.0],
            "alts": [500.0, 500.0],
            "vxs": [100.0, 100.0],
            "vys": [200.0, 200.0],
            "vzs": [300.0, 300.0],
            "cross_section_model": {"rcs": 1.0},
        }

        trajectory = Trajectory.model_validate(data)

        self.assertNotIn("vxs", trajectory.model_dump())
        target = trajectory(T0)
        self.assertEqual(target.velocity.as_tuple(), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
