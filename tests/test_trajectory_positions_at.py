import datetime
import unittest

import numpy as np

from theia.types import ConstantRcsModel, TargetInfos, Trajectory


def _trajectory(
    t0: datetime.datetime,
    lats: list[float],
    lons: list[float],
    alts: list[float],
    offsets_s: list[float] | None = None,
) -> Trajectory:
    """Waypoints at `t0` plus `offsets_s` [s]; 10 s apart if not given."""
    n = len(lats)
    if offsets_s is None:
        offsets_s = [10.0 * i for i in range(n)]
    return Trajectory(
        target_id=0,
        target_info=TargetInfos.FIXED_WING,
        times=[t0 + datetime.timedelta(seconds=s) for s in offsets_s],
        lats=lats,
        lons=lons,
        alts=alts,
        vxs=[0.0] * n,
        vys=[0.0] * n,
        vzs=[0.0] * n,
        cross_section_model=ConstantRcsModel(rcs=1.0),
    )


class TrajectoryPositionsAtTest(unittest.TestCase):
    def setUp(self):
        t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
        self.trajectory = _trajectory(
            t0,
            lats=[47.0, 47.1, 47.15, 47.3, 47.32],
            lons=[8.0, 8.05, 8.2, 8.21, 8.4],
            alts=[100.0, 500.0, 450.0, 1_000.0, 300.0],
        )
        self.timestamps = [t.timestamp() for t in self.trajectory.times]

    def test_shape(self):
        positions = self.trajectory.positions_at(np.linspace(0.0, 40.0, 7))
        self.assertEqual(positions.shape, (7, 3))

    def test_empty_input(self):
        positions = self.trajectory.positions_at(np.array([]))
        self.assertEqual(positions.shape, (0, 3))

    def test_waypoints(self):
        positions = self.trajectory.positions_at(np.array(self.timestamps))
        np.testing.assert_allclose(positions[:, 0], self.trajectory.lats, rtol=1e-12)
        np.testing.assert_allclose(positions[:, 1], self.trajectory.lons, rtol=1e-12)
        np.testing.assert_allclose(positions[:, 2], self.trajectory.alts, rtol=1e-12)

    def test_two_waypoints_interpolate_linearly(self):
        t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
        trajectory = _trajectory(
            t0, lats=[47.0, 47.1], lons=[8.0, 8.2], alts=[100.0, 300.0]
        )
        positions = trajectory.positions_at(np.array([2.5, 5.0]))
        np.testing.assert_allclose(
            positions, [[47.025, 8.05, 150.0], [47.05, 8.1, 200.0]], rtol=1e-12
        )

    def test_identical_to_call(self):
        # positions_at and __call__ must agree exactly, not just approximately:
        # callers may switch between them, e.g. trajectory_intersects_terrain.
        for t0 in (
            datetime.datetime(2026, 10, 30, tzinfo=datetime.UTC),
            datetime.datetime(2026, 10, 30),  # naive, i.e. local time
        ):
            with self.subTest(t0=t0):
                trajectory = _trajectory(
                    t0,
                    lats=self.trajectory.lats,
                    lons=self.trajectory.lons,
                    alts=self.trajectory.alts,
                )
                times = [
                    t0 + datetime.timedelta(seconds=s)
                    for s in np.linspace(0.0, 40.0, 101).tolist()
                ]
                positions = trajectory.positions_at(
                    np.array([t.timestamp() for t in times])
                )
                for t, position in zip(times, positions.tolist()):
                    target = trajectory(t)
                    self.assertEqual(position, [target.lat, target.lon, target.alt])

    def test_unequal_spacing(self):
        # Gaps from 0.5 s to 100 s: each time must fall into the right piece.
        t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
        offsets_s = [0.0, 1.0, 101.0, 101.5, 131.5]
        trajectory = _trajectory(
            t0,
            lats=self.trajectory.lats,
            lons=self.trajectory.lons,
            alts=self.trajectory.alts,
            offsets_s=offsets_s,
        )
        positions = trajectory.positions_at(np.array(offsets_s))
        np.testing.assert_allclose(positions[:, 0], trajectory.lats, rtol=1e-12)
        np.testing.assert_allclose(positions[:, 1], trajectory.lons, rtol=1e-12)
        np.testing.assert_allclose(positions[:, 2], trajectory.alts, rtol=1e-12)

        times = [
            t0 + datetime.timedelta(seconds=s)
            for s in np.linspace(0.0, offsets_s[-1], 101).tolist()
        ]
        positions = trajectory.positions_at(np.array([t.timestamp() for t in times]))
        for t, position in zip(times, positions.tolist()):
            target = trajectory(t)
            self.assertEqual(position, [target.lat, target.lon, target.alt])

    def test_outside_is_nan_like_call_is_none(self):
        before = self.trajectory.times[0] - datetime.timedelta(milliseconds=1)
        after = self.trajectory.times[-1] + datetime.timedelta(milliseconds=1)
        positions = self.trajectory.positions_at(
            np.array([before.timestamp(), after.timestamp()])
        )
        self.assertTrue(np.isnan(positions).all())
        self.assertIsNone(self.trajectory(before))
        self.assertIsNone(self.trajectory(after))

    def test_endpoints_are_inside(self):
        first = self.trajectory.times[0]
        last = self.trajectory.times[-1]
        positions = self.trajectory.positions_at(
            np.array([first.timestamp(), last.timestamp()])
        )
        self.assertFalse(np.isnan(positions).any())
        for t, position in zip((first, last), positions.tolist()):
            target = self.trajectory(t)
            self.assertIsNotNone(target)
            self.assertEqual(position, [target.lat, target.lon, target.alt])

    def test_mixed_inside_and_outside(self):
        timestamps = np.array(
            [self.timestamps[0] - 5.0, self.timestamps[2], self.timestamps[-1] + 5.0]
        )
        positions = self.trajectory.positions_at(timestamps)
        self.assertTrue(np.isnan(positions[0]).all())
        self.assertFalse(np.isnan(positions[1]).any())
        self.assertTrue(np.isnan(positions[2]).all())
        self.assertAlmostEqual(positions[1, 0], self.trajectory.lats[2], places=12)
        self.assertAlmostEqual(positions[1, 1], self.trajectory.lons[2], places=12)
        self.assertAlmostEqual(positions[1, 2], self.trajectory.alts[2], places=12)


if __name__ == "__main__":
    unittest.main()
