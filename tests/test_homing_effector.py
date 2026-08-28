import datetime
import math
import unittest
from unittest.mock import MagicMock, patch

from theia.config import SIDC
from theia.coordinates import CoordinateTransformations
from theia.effectors import DirectFireEffector
from theia.simulation.controllers.homing_effector import HomingSystem
from theia.types import ConstantRcsModel, DirectShot, KillEvent, Point, Target


def make_track(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0):
    """
    Build a fake 'track' callable.

    In the real code, `track(t)` returns (x, vx, y, vy, z, vz) at time t.
    Here we return a constant (non-moving) target regardless of t, which
    is enough for most branch tests. Tests that care about motion can
    build their own callable.
    """

    def _track(_t):
        return (x, vx, y, vy, z, vz)

    return _track


class HomingSystemTestCase(unittest.TestCase):
    """Base class with a helper to build a HomingSystem with mocked collaborators."""

    def make_system(self, **overrides):
        defaults = dict(
            target_id=1,
            sidc=SIDC.BLUE_MISSILE,
            speed=100.0,
            max_dist=10_000.0,
            point=Point(lat=0.0, lon=0.0, alt=0.0) if Point else MagicMock(),
            rcs=ConstantRcsModel(rcs=1.0),
            effector=MagicMock(name="effector", id=42, combat_range=5_000.0),
            assigned_track_id=99,
            terrain=MagicMock(name="terrain"),
            name="TestHomer",
            travelled_dist=0.0,
        )
        defaults.update(overrides)
        system = HomingSystem.model_construct(**defaults)
        # Broadcasting is presumably provided by a base class (Controller);
        # mock it directly so we can assert on it without needing the
        # full Controller machinery.
        system._broadcast_event = MagicMock(name="_broadcast_event")
        return system


class TestOnEvent(HomingSystemTestCase):
    def test_shot_from_own_effector_triggers_suicide(self):
        system = self.make_system()
        system._commit_suicide = MagicMock()
        event = DirectShot.model_construct(
            id=0,
            time=datetime.datetime(2026, 1, 1),
            shooter=MagicMock(id=42),
            target=MagicMock(id=2),
        )
        # Ensure event.shooter.id matches system.effector.id
        system.on_event(event)
        system._commit_suicide.assert_called_once_with(event.time)

    def test_shot_from_other_effector_does_not_trigger_suicide(self):
        system = self.make_system()
        system._commit_suicide = MagicMock()
        event = DirectShot.model_construct(
            id=0,
            time=datetime.datetime(2026, 1, 1),
            shooter=MagicMock(id=999),
            target=MagicMock(id=2),
        )
        system.on_event(event)

        system._commit_suicide.assert_not_called()

    def test_non_shot_event_is_ignored(self):
        system = self.make_system()
        system._commit_suicide = MagicMock()

        system.on_event(object())

        system._commit_suicide.assert_not_called()


class TestCommitSuicide(HomingSystemTestCase):
    def test_broadcasts_kill_event_with_correct_target(self):
        system = self.make_system()
        t = datetime.datetime(2026, 1, 1, 12, 0, 0)

        system._commit_suicide(t)

        self.assertEqual(system._broadcast_event.call_count, 1)
        broadcast_arg = system._broadcast_event.call_args[0][0]
        self.assertIsInstance(broadcast_arg, KillEvent)
        self.assertEqual(broadcast_arg.time, t)
        self.assertEqual(broadcast_arg.target_id, 1)


class TestGetNextPosition(HomingSystemTestCase):
    def test_out_of_fuel_commits_suicide(self):
        system = self.make_system(max_dist=100.0, travelled_dist=99.0, speed=1.0)
        system._commit_suicide = MagicMock()
        sp = MagicMock(time=datetime.datetime(2026, 1, 1), enemy_targets=[])
        dt = datetime.timedelta(seconds=10)  # seconds_left=1 < dt.seconds=10

        result = system._get_next_position(sp, dt)

        system._commit_suicide.assert_called_once()
        self.assertIsNone(result)

    def test_missing_track_commits_suicide(self):
        system = self.make_system(assigned_track_id=99)
        system._commit_suicide = MagicMock()
        sp = MagicMock(
            time=datetime.datetime(2026, 1, 1),
            enemy_targets=[MagicMock(id=1)],  # no track with id==99
        )
        dt = datetime.timedelta(seconds=1)

        result = system._get_next_position(sp, dt)

        system._commit_suicide.assert_called_once()
        self.assertIsNone(result)

    @patch("theia.simulation.controllers.homing_effector")
    def test_normal_case_returns_new_point(self, mock_minimize_scalar):
        # Arrange a stationary track far along the x-axis.
        track = make_track(x=1000.0, y=0.0, z=0.0)
        track.id = 99
        system = self.make_system(assigned_track_id=99, speed=50.0)
        system._commit_suicide = MagicMock()

        sp = MagicMock(time=datetime.datetime(2026, 1, 1))
        sp.enemy_targets = [track]
        dt = datetime.timedelta(seconds=1)

        mock_minimize_scalar.return_value = MagicMock(x=[5.0])

        result = system._get_next_position(sp, dt)

        system._commit_suicide.assert_not_called()
        self.assertIsInstance(result, Point)


class TestGetTargets(HomingSystemTestCase):
    def test_returns_single_target_with_expected_fields(self):
        system = self.make_system(target_id=7, name="Interceptor-1")

        targets = system._get_targets()

        self.assertEqual(len(targets), 1)
        target = targets[0]
        self.assertIsInstance(target, Target)
        self.assertEqual(target.id, 7)
        self.assertEqual(target.name, "Interceptor-1")
        self.assertFalse(target.is_stationary)
        self.assertEqual(target.point, system.point)
        self.assertEqual(target.cross_section_model, system.rcs)


class TestGetFiringEffectors(HomingSystemTestCase):
    def _sp_with_track(self, track, track_id=99):
        track.id = track_id
        sp = MagicMock(time=datetime.datetime(2026, 1, 1))
        sp.enemy_targets = [track]
        return sp

    def test_no_matching_track_returns_empty(self):
        system = self.make_system(assigned_track_id=99)
        sp = MagicMock(time=datetime.datetime(2026, 1, 1))
        sp.enemy_targets = [MagicMock(id=1)]
        dt = datetime.timedelta(seconds=1)

        result = system._get_firing_effectors(sp, dt)

        self.assertEqual(result, [])

    @patch("theia.simulation.controllers.homing_effector.line_of_sight_distance")
    def test_in_range_with_los_returns_effector_and_point(self, mock_los_distance):
        track = make_track(x=0.0, y=0.0, z=0.0)
        system = self.make_system(assigned_track_id=99)
        system.effector.combat_range = 5000.0
        system.terrain.has_line_of_sight.return_value = True
        mock_los_distance.return_value = 1000.0  # within combat_range
        sp = self._sp_with_track(track)
        dt = datetime.timedelta(seconds=1)

        result = system._get_firing_effectors(sp, dt)

        self.assertEqual(len(result), 1)
        effector, point = result[0]
        self.assertEqual(effector, system.effector)
        self.assertEqual(system.effector.point, system.point)

    @patch("theia.simulation.controllers.homing_effector.line_of_sight_distance")
    def test_out_of_range_returns_empty(self, mock_los_distance):
        track = make_track()
        system = self.make_system(assigned_track_id=99)
        system.effector.combat_range = 100.0
        mock_los_distance.return_value = 5000.0  # beyond combat_range
        sp = self._sp_with_track(track)
        dt = datetime.timedelta(seconds=1)

        result = system._get_firing_effectors(sp, dt)

        self.assertEqual(result, [])

    @patch("theia.simulation.controllers.homing_effector.line_of_sight_distance")
    def test_in_range_but_no_los_returns_empty(self, mock_los_distance):
        track = make_track()
        system = self.make_system(assigned_track_id=99)
        system.effector.combat_range = 5000.0
        system.terrain.has_line_of_sight.return_value = False
        mock_los_distance.return_value = 1000.0
        sp = self._sp_with_track(track)
        dt = datetime.timedelta(seconds=1)

        with patch.object(
            CoordinateTransformations,
            "cartesian_to_geodetic",
            return_value=(1.0, 2.0, 3.0),
        ):
            result = system._get_firing_effectors(sp, dt)

        self.assertEqual(result, [])


class TestUpdate(HomingSystemTestCase):
    def test_update_sets_point_targets_and_firing_effectors(self):
        system = self.make_system()
        new_point = Point(lat=12, lon=34, alt=56)
        targets = [Target.model_construct(name="target")]
        firing_effectors = [
            (DirectFireEffector.model_construct(), Point.model_construct()),
        ]

        system._get_next_position = MagicMock(return_value=new_point)
        system._get_targets = MagicMock(return_value=targets)
        system._get_firing_effectors = MagicMock(return_value=firing_effectors)

        sp = MagicMock(time=datetime.datetime(2026, 1, 1))
        dt = datetime.timedelta(seconds=1)

        system.update(sp, dt)

        self.assertEqual(system.point, new_point)
        self.assertEqual(system.targets, targets)
        self.assertEqual(system.firing_effectors, firing_effectors)
        system._get_next_position.assert_called_once_with(sp, dt)
        system._get_targets.assert_called_once_with()
        system._get_firing_effectors.assert_called_once_with(sp, dt)

    def test_update_advances_travelled_dist_by_actual_distance_moved(self):
        # Target is 5 units away; at speed=10 over dt=1s the interceptor
        # could cover 10 units, but must stop exactly on the target (5
        # units) rather than overshoot. travelled_dist must reflect the
        # actual distance moved (5), not the nominal dt * speed (10).
        with (
            patch.object(
                CoordinateTransformations,
                "geodetic_to_cartesian",
                side_effect=lambda lat, lon, alt: (lat, lon, alt),
            ),
            patch.object(
                CoordinateTransformations,
                "cartesian_to_geodetic",
                side_effect=lambda x, y, z: (x, y, z),
            ),
        ):
            track = make_track(x=5.0, y=0.0, z=0.0)
            track.id = 99
            system = self.make_system(
                assigned_track_id=99,
                speed=10.0,
                max_dist=1000.0,
                travelled_dist=0.0,
                point=Point(lat=0.0, lon=0.0, alt=0.0),
            )
            system._get_firing_effectors = MagicMock(return_value=[])
            sp = MagicMock(time=datetime.datetime(2026, 1, 1))
            sp.enemy_targets = [track]
            dt = datetime.timedelta(seconds=1)

            system.update(sp, dt)

        self.assertAlmostEqual(system.travelled_dist, 5.0)


class TestGetNextPositionGeometry(HomingSystemTestCase):
    """
    Tests to verify closest-point-of-approach and overshoot-avoidance behavior.

    We patch CoordinateTransformations to an identity mapping so we can reason
    about positions as plain (x, y, z) tuples instead of real geodetic math.
    """

    def setUp(self):
        self.g2c_patch = patch.object(
            CoordinateTransformations,
            "geodetic_to_cartesian",
            side_effect=lambda lat, lon, alt: (lat, lon, alt),
        )
        self.c2g_patch = patch.object(
            CoordinateTransformations,
            "cartesian_to_geodetic",
            side_effect=lambda x, y, z: (x, y, z),
        )
        self.g2c_patch.start()
        self.c2g_patch.start()
        self.addCleanup(self.g2c_patch.stop)
        self.addCleanup(self.c2g_patch.stop)

    def test_reaches_stationary_target_within_range(self):
        # Target is 5 units away, well within reach at speed=10 over up to
        # 100 seconds. Expect the system to land exactly on the target,
        # not fly past it.
        track = make_track(x=5.0, y=0.0, z=0.0)
        track.id = 99
        system = self.make_system(
            assigned_track_id=99,
            speed=10.0,
            max_dist=1000.0,
            travelled_dist=0.0,
            point=Point(lat=0.0, lon=0.0, alt=0.0),
        )
        sp = MagicMock(time=datetime.datetime(2026, 1, 1))
        sp.enemy_targets = [track]
        dt = datetime.timedelta(seconds=1)

        result = system._get_next_position(sp, dt)

        self.assertIsInstance(result, Point)
        self.assertAlmostEqual(result.lat, 5.0, places=3)
        self.assertAlmostEqual(result.lon, 0.0, places=3)
        self.assertAlmostEqual(result.alt, 0.0, places=3)

    def test_does_not_overshoot_unreachable_target(self):
        # Target is 1000 units away. At speed=10 with only 2 seconds of
        # "seconds_left" budget, the system can travel at most 20 units.
        # It must move the full 20 units toward the target and stop there
        # -- not overshoot, and not fall short of its own max reach.
        track = make_track(x=1000.0, y=0.0, z=0.0)
        track.id = 99
        system = self.make_system(
            assigned_track_id=99,
            speed=10.0,
            max_dist=20.0,  # seconds_left = (20 - 0) / 10 = 2s
            travelled_dist=0.0,
            point=Point(lat=0.0, lon=0.0, alt=0.0),
        )
        sp = MagicMock(time=datetime.datetime(2026, 1, 1))
        sp.enemy_targets = [track]
        dt = datetime.timedelta(
            seconds=1
        )  # seconds_left(2) >= dt.seconds(1), no suicide

        result = system._get_next_position(sp, dt)

        distance_travelled = math.sqrt(result.lat**2 + result.lon**2 + result.alt**2)
        distance_to_target = math.sqrt(
            (1000.0 - result.lat) ** 2 + result.lon**2 + result.alt**2
        )

        # Travelled (close to) the full reachable distance and did not
        # pass the target.
        self.assertAlmostEqual(distance_travelled, 10.0)
        self.assertAlmostEqual(distance_to_target, 990.0)
        self.assertAlmostEqual(
            distance_travelled + distance_to_target,
            1000.0,
        )

    def test_leads_a_moving_target_rather_than_chasing_current_position(self):
        dt = datetime.timedelta(seconds=1)

        # Target moving away in +x at 20 units/s, starting 100 units out.
        # If the system merely aimed at the target's *current* position each
        # tick it would fall further behind. With enough speed/time budget it
        # should instead solve for a future intercept point.
        def ground_truth(t):
            # t is an absolute datetime; convert to elapsed seconds from
            # situational_picture.time for a simple linear model.
            elapsed = (t - datetime.datetime(2026, 1, 1)).seconds
            x = 100.0 + 20.0 * elapsed
            return (x, 20.0, 0.0, 0.0, 0.0, 0.0)

        ground_truth.id = 99
        system = self.make_system(
            assigned_track_id=99,
            speed=50.0,  # faster than the target
            max_dist=100_000.0,  # plenty of fuel budget
            travelled_dist=0.0,
            point=Point(lat=0.0, lon=0.0, alt=0.0),
        )
        sp = MagicMock(time=datetime.datetime(2026, 1, 1))
        sp.enemy_targets = [ground_truth]
        

        for _ in range(3):
            result = system._get_next_position(sp, dt)
            system.point = result
            self.assertIsNotNone(result)
            px, py, pz = CoordinateTransformations.geodetic_to_cartesian(
                result.lat,
                result.lon,
                result.alt,
            )
            x, vx, y, vy, z, vz = ground_truth(sp.time + dt)
            d2 = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2
            print(math.sqrt(d2))
            self.assertGreater(d2, 0)
            sp.time += dt

        print("===================================")
        result = system._get_next_position(sp, dt)
        self.assertIsNotNone(result)
        system.point = result
        px, py, pz = CoordinateTransformations.geodetic_to_cartesian(
            result.lat,
            result.lon,
            result.alt,
        )
        x, vx, y, vy, z, vz = ground_truth(sp.time + dt)
        d2 = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2
        print(math.sqrt(d2))
        self.assertAlmostEqual(d2, 0)


if __name__ == "__main__":
    unittest.main()
