import datetime
import unittest

import numpy as np

from theia.config import INTEL_DETECTION_SIGMA
from theia.simulation.trackers.pseudo_tracker import PseudoTracker
from theia.types import (
    ConstantRcsModel,
    IdProvider,
    IntelDetection,
    Party,
    TargetInfos,
    Track,
    Trajectory,
)

T0 = datetime.datetime.fromtimestamp(1_000_000, datetime.UTC)
N_DETECTIONS = 200
"""One detection per second"""


def make_trajectory() -> Trajectory:
    """Northbound flight at 100 m/s for the duration of the detections."""
    dts = np.linspace(0.0, N_DETECTIONS, 11)
    return Trajectory(
        target_id=7,
        target_info=TargetInfos.FIXED_WING.with_party(Party.RED),
        times=[T0 + datetime.timedelta(seconds=float(dt)) for dt in dts],
        lats=[47.0 + 100.0 * dt / 111_000.0 for dt in dts],
        lons=[8.0] * len(dts),
        alts=[3000.0] * len(dts),
        cross_section_model=ConstantRcsModel(rcs=1.0),
    )


def make_tracker(track_history: datetime.timedelta) -> PseudoTracker:
    # The same seed for every tracker: identical detection noise.
    return PseudoTracker(
        removal_patience=5,
        rng=np.random.Generator(np.random.PCG64(seed=3)),
        start_time=T0,
        track_history=track_history,
    )


def detect(tracker: PseudoTracker, trajectory: Trajectory, i: int):
    t = T0 + datetime.timedelta(seconds=i)
    detection = IntelDetection(
        detection_id=i, time=t, target=trajectory(t), sigma=INTEL_DETECTION_SIGMA
    )
    tracker.add_detections([], [], [], [], [detection], IdProvider())


def tracked(track_history: datetime.timedelta) -> Track:
    trajectory = make_trajectory()
    tracker = make_tracker(track_history)
    for i in range(N_DETECTIONS):
        detect(tracker, trajectory, i)
    [track] = tracker.get_tracks()
    return track


class TrackHistoryTest(unittest.TestCase):
    def test_window_yields_the_same_states_as_the_full_track(self):
        # 60.5 s: the window does not start on a state.
        history = datetime.timedelta(seconds=60.5)
        windowed = tracked(history)
        full = tracked(datetime.timedelta(days=1))
        self.assertEqual(len(full.states), N_DETECTIONS)
        # Last state at 199 s, window from 138.5 s: the states from 138 s (the
        # last one at or before the start of the window) to 199 s.
        self.assertEqual(len(windowed.states), 62)
        self.assertEqual(windowed.states[0][0], T0 + datetime.timedelta(seconds=138))

        t_last = windowed.states[-1][0]
        # From the start of the window to past the end (extrapolation and
        # inactive time).
        for seconds in np.arange(-60.5, 45.0, 0.25):
            t = t_last + datetime.timedelta(seconds=float(seconds))
            np.testing.assert_allclose(windowed(t), full(t), rtol=0, atol=1e-9)

    def test_window_keeps_at_least_two_states(self):
        track = tracked(datetime.timedelta(0))
        self.assertEqual(len(track.states), 2)

    def assert_same_track(self, actual: Track, expected: Track):
        self.assertEqual(actual.id, expected.id)
        self.assertEqual(actual.target_info, expected.target_info)
        self.assertEqual([t for t, _ in actual.states], [t for t, _ in expected.states])
        np.testing.assert_array_equal(
            [state for _, state in actual.states],
            [state for _, state in expected.states],
        )

    def test_repeated_export_is_equal(self):
        trajectory = make_trajectory()
        tracker = make_tracker(datetime.timedelta(seconds=60))
        for i in range(3):
            detect(tracker, trajectory, i)
        [first] = tracker.get_tracks()
        [again] = tracker.get_tracks()
        self.assert_same_track(again, first)

    def test_export_reflects_new_detections(self):
        trajectory = make_trajectory()
        tracker = make_tracker(datetime.timedelta(seconds=60))
        for i in range(3):
            detect(tracker, trajectory, i)
        tracker.get_tracks()
        detect(tracker, trajectory, 3)
        [updated] = tracker.get_tracks()

        # Same detections, but exported only at the end.
        reference = make_tracker(datetime.timedelta(seconds=60))
        for i in range(4):
            detect(reference, trajectory, i)
        [expected] = reference.get_tracks()

        self.assert_same_track(updated, expected)
        self.assertEqual(updated.states[-1][0], T0 + datetime.timedelta(seconds=3))


if __name__ == "__main__":
    unittest.main()
