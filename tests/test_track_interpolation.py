import datetime
import unittest

import numpy as np
from scipy.interpolate import make_interp_spline

from theia.types import TargetInfos, Track

T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
INACTIVE = datetime.timedelta(seconds=30)
# Irregular time steps [s]
OFFSETS = [0.0, 1.0, 2.5, 3.0, 7.0]


def make_states() -> list[tuple[datetime.datetime, np.ndarray]]:
    rng = np.random.default_rng(seed=1)
    return [
        (T0 + datetime.timedelta(seconds=dt), rng.normal(0, 1000, 6)) for dt in OFFSETS
    ]


def make_track() -> Track:
    return Track(
        id="1",
        target_info=TargetInfos.FIXED_WING,
        states=make_states(),
        inactive_time=INACTIVE,
    )


def query_offsets() -> np.ndarray:
    """Before, at and between the states, and after the last one."""
    return np.concatenate([np.arange(-3.0, 45.0, 0.25), OFFSETS])


class TrackInterpolationTest(unittest.TestCase):
    def test_matches_linear_spline(self):
        """Same values as the previous implementation (scipy, k=1)."""
        states = make_states()
        reference = make_interp_spline(
            [t.timestamp() for t, _ in states], np.array([y for _, y in states]), k=1
        )
        t_last = states[-1][0].timestamp()
        track = make_track()
        for dt in query_offsets():
            time = T0 + datetime.timedelta(seconds=float(dt))
            # Capped at inactive_time after the last state.
            t = min(time.timestamp(), t_last + INACTIVE.total_seconds())
            np.testing.assert_allclose(track(time), reference(t), rtol=1e-12, atol=1e-9)

    def test_from_arrays_equals_validated_track(self):
        states = make_states()
        expected = make_track()
        actual = Track.from_arrays(
            id="1",
            target_info=TargetInfos.FIXED_WING,
            times=[t for t, _ in states],
            timestamps=np.array([t.timestamp() for t, _ in states]),
            y=np.array([y for _, y in states]),
            inactive_time=INACTIVE,
        )
        self.assertEqual(
            actual.model_dump(mode="json"), expected.model_dump(mode="json")
        )
        for dt in query_offsets():
            time = T0 + datetime.timedelta(seconds=float(dt))
            np.testing.assert_array_equal(actual(time), expected(time))

    def test_duplicate_times_raise(self):
        states = make_states()
        states[2] = (states[1][0], states[2][1])
        with self.assertRaises(ValueError):
            Track(id="1", target_info=TargetInfos.FIXED_WING, states=states)

    def test_single_state_raises(self):
        with self.assertRaises(ValueError):
            Track(id="1", target_info=TargetInfos.FIXED_WING, states=make_states()[:1])


if __name__ == "__main__":
    unittest.main()
