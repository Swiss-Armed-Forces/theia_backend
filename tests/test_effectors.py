import datetime
import unittest

from theia.config import SIDC, UNKNOWN_ID, UNKNOWN_TIME
from theia.coordinates import POSITIONS_OF_INTEREST
from theia.effectors import (
    DirectFireEffector,
    IndirectFireEffector,
    NoLosException,
    OnCooldownException,
    OutOfAttacksException,
    OutOfRangeException,
    TooManyInFlightException,
)
from theia.terrain import SrtmTerrainModel
from theia.types import (
    ConstantRcsModel,
    DirectShot,
    IndirectShot,
    Point,
    Target,
    Velocity,
)

srtm = SrtmTerrainModel()

t0 = datetime.datetime.fromtimestamp(0)

p_uetliberg = Point(
    lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
    lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
    alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
)

p_bern = Point(
    lat=46.948056,
    lon=7.4475,
    alt=1000.0,
)


p_close = Point(
    lat=47.37348,
    lon=8.53707,
    alt=1000.0,
)

p_no_los = Point(
    lat=47.22726,
    lon=8.66719,
    alt=srtm.elevationAt(47.22726, 8.66719),
)


class DirectEffectorTest(unittest.TestCase):
    # def test_no_assigned_target(self):
    #     raise NotImplementedError()

    def _make_effector(self, **overrides) -> DirectFireEffector:
        defaults = {
            "id": 0,
            "point": p_uetliberg,
            "combat_range": 100_000,
            "n_attacks_left": 1,
            "name": "",
            "terrain": srtm,
            # Unlimited rate of fire by default - these tests aren't about
            # cadence; see CadenceTest for that.
            "cadence": float("inf"),
        }
        defaults.update(overrides)
        return DirectFireEffector(**defaults)

    def test_too_far_away(self):
        effector = self._make_effector(combat_range=10_000)
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_bern,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(OutOfRangeException):
            effector.fire(target, t0)

    def test_no_attacks_left(self):
        effector = self._make_effector(combat_range=10_000, n_attacks_left=0)
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(OutOfAttacksException):
            effector.fire(target, t0)

    def test_no_los(self):
        effector = self._make_effector()
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_no_los,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(NoLosException):
            effector.fire(target, t0)

    def test_use_ammo(self):
        effector = self._make_effector()
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        self.assertEqual(effector.n_attacks_left, 1)
        effector.fire(target, t0)
        self.assertEqual(effector.n_attacks_left, 0)

        with self.assertRaises(OutOfAttacksException):
            effector.fire(target, t0)

    def test_result(self):
        effector = self._make_effector()
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        event = effector.fire(target, t0)
        self.assertIsInstance(event, DirectShot)
        self.assertEqual(event.target, target)
        self.assertEqual(event.shooter, effector)
        self.assertEqual(event.id, UNKNOWN_ID)
        self.assertEqual(event.time, UNKNOWN_TIME)


class IndirectEffectorTest(unittest.TestCase):
    def setUp(self):
        self._projectile = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
            cadence=float("inf"),
        )

    def _make_effector(self, **overrides) -> IndirectFireEffector:
        defaults = {
            "id": 0,
            "point": p_uetliberg,
            "combat_range": 100_000,
            "n_attacks_left": 1,
            "name": "",
            "projectile": self._projectile,
            "projectile_speed": 300.0,
            "projectile_max_dist": 50_000.0,
            "projectile_sidc": SIDC.BLUE_MISSILE,
            "projectile_rcs": ConstantRcsModel(rcs=0.1),
            "assigned_track_id": "0",
            # Unlimited rate of fire by default - these tests aren't about
            # cadence; see CadenceTest for that.
            "cadence": float("inf"),
            # Effectively unlimited in-flight cap - these tests aren't
            # about the in-flight cap; see InFlightCapTest for that.
            "max_in_flight": 1_000_000,
        }
        defaults.update(overrides)
        return IndirectFireEffector(**defaults)

    def test_too_far_away(self):
        effector = self._make_effector(combat_range=10_000)
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_bern,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(OutOfRangeException):
            effector.fire(target, t0)

    def test_no_attacks_left(self):
        effector = self._make_effector(combat_range=10_000, n_attacks_left=0)
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(OutOfAttacksException):
            effector.fire(target, t0)

    def test_los_irrelevant(self):
        effector = self._make_effector()
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_no_los,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        effector.fire(target, t0)

    def test_use_ammo(self):
        effector = self._make_effector()
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        self.assertEqual(effector.n_attacks_left, 1)
        effector.fire(target, t0)
        self.assertEqual(effector.n_attacks_left, 0)

        with self.assertRaises(OutOfAttacksException):
            effector.fire(target, t0)

    def test_indirect_fire_event(self):
        # Deliberately give the target a different (ground-truth) ID than the
        # assigned track ID: a track can correspond to multiple targets or to
        # clutter, so the two must never be conflated.
        effector = self._make_effector(assigned_track_id="7")
        target = Target(
            id=99,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        event = effector.fire(target, t0)
        self.assertIsInstance(event, IndirectShot)
        self.assertEqual(event.target, target)
        self.assertEqual(event.shooter, effector)
        self.assertEqual(event.id, UNKNOWN_ID)
        self.assertEqual(event.time, UNKNOWN_TIME)
        self.assertEqual(event.projectile, self._projectile)
        self.assertEqual(event.track_id, "7")


class CadenceTest(unittest.TestCase):
    """
    Cadence is shared behaviour between DirectFireEffector and
    IndirectFireEffector (both inherit it from AbstractEffector / fire()'s
    shared _check_cadence helper), so both are covered here.
    """

    def _make_direct_effector(self, **overrides) -> DirectFireEffector:
        defaults = {
            "id": 0,
            "point": p_uetliberg,
            "combat_range": 100_000,
            "n_attacks_left": 5,
            "name": "",
            "terrain": srtm,
            "cadence": 1.0,
        }
        defaults.update(overrides)
        return DirectFireEffector(**defaults)

    def _make_indirect_effector(self, **overrides) -> IndirectFireEffector:
        projectile = DirectFireEffector(
            id=1,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
            cadence=float("inf"),
        )
        defaults = {
            "id": 0,
            "point": p_uetliberg,
            "combat_range": 100_000,
            "n_attacks_left": 5,
            "name": "",
            "projectile": projectile,
            "projectile_speed": 300.0,
            "projectile_max_dist": 50_000.0,
            "projectile_sidc": SIDC.BLUE_MISSILE,
            "projectile_rcs": ConstantRcsModel(rcs=0.1),
            "assigned_track_id": "0",
            "cadence": 1.0,
            # Effectively unlimited in-flight cap - these tests are about
            # cadence, not the in-flight cap; see InFlightCapTest for that.
            "max_in_flight": 1_000_000,
        }
        defaults.update(overrides)
        return IndirectFireEffector(**defaults)

    def _target(self) -> Target:
        return Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

    def test_first_shot_never_blocked(self):
        for effector in (self._make_direct_effector(), self._make_indirect_effector()):
            with self.subTest(effector=type(effector).__name__):
                effector.fire(self._target(), t0)  # must not raise

    def test_second_shot_within_cadence_window_is_blocked(self):
        # cadence=1.0 => must wait 1s between shots.
        for effector in (self._make_direct_effector(), self._make_indirect_effector()):
            with self.subTest(effector=type(effector).__name__):
                effector.fire(self._target(), t0)
                with self.assertRaises(OnCooldownException):
                    effector.fire(self._target(), t0 + datetime.timedelta(seconds=0.5))

    def test_shot_allowed_again_once_cadence_window_elapses(self):
        for effector in (self._make_direct_effector(), self._make_indirect_effector()):
            with self.subTest(effector=type(effector).__name__):
                effector.fire(self._target(), t0)
                # Exactly one cadence period later - must not be blocked.
                effector.fire(self._target(), t0 + datetime.timedelta(seconds=1))

    def test_cadence_only_consumed_by_a_confirmed_shot(self):
        # A failed attempt (out of range here) must not itself start a
        # cooldown - only a confirmed fire() success should.
        effector = self._make_direct_effector(combat_range=10_000)
        far_target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_bern,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )
        with self.assertRaises(OutOfRangeException):
            effector.fire(far_target, t0)

        self.assertIsNone(effector.time_of_last_shot)
        # Immediately afterwards, a shot at an in-range target must succeed -
        # the failed attempt above must not have put it on cooldown.
        effector.fire(self._target(), t0)


class InFlightCapTest(unittest.TestCase):
    """
    IndirectFireEffector.max_in_flight caps how many of its own projectiles
    may be in the air at once, independently of cadence: a fast cadence must
    not let it launch a new one while an earlier one hasn't been resolved
    yet (hit, lost track, or ran out of fuel - see HomingSystem).
    """

    def _make_effector(self, **overrides) -> IndirectFireEffector:
        projectile = DirectFireEffector(
            id=1,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
            cadence=float("inf"),
        )
        defaults = {
            "id": 0,
            "point": p_uetliberg,
            "combat_range": 100_000,
            "n_attacks_left": 5,
            "name": "",
            "projectile": projectile,
            "projectile_speed": 300.0,
            "projectile_max_dist": 50_000.0,
            "projectile_sidc": SIDC.BLUE_MISSILE,
            "projectile_rcs": ConstantRcsModel(rcs=0.1),
            "assigned_track_id": "0",
            # Unlimited cadence - these tests aren't about cadence.
            "cadence": float("inf"),
        }
        defaults.update(overrides)
        return IndirectFireEffector(**defaults)

    def _target(self) -> Target:
        return Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

    def test_default_max_in_flight_is_one(self):
        effector = self._make_effector()
        self.assertEqual(effector.max_in_flight, 1)

    def test_fire_increments_n_in_flight(self):
        effector = self._make_effector()
        self.assertEqual(effector.n_in_flight, 0)
        effector.fire(self._target(), t0)
        self.assertEqual(effector.n_in_flight, 1)

    def test_second_launch_blocked_while_first_still_in_flight(self):
        effector = self._make_effector(max_in_flight=1)
        effector.fire(self._target(), t0)

        with self.assertRaises(TooManyInFlightException):
            effector.fire(self._target(), t0)

    def test_launch_allowed_again_once_in_flight_count_drops(self):
        effector = self._make_effector(max_in_flight=1)
        effector.fire(self._target(), t0)

        # Simulate the in-flight projectile being resolved (hit / lost
        # track / out of fuel), as HomingSystem._commit_suicide would do.
        effector.n_in_flight -= 1

        effector.fire(self._target(), t0)  # must not raise

    def test_max_in_flight_greater_than_one_allows_concurrent_launches(self):
        effector = self._make_effector(max_in_flight=2)

        effector.fire(self._target(), t0)
        effector.fire(self._target(), t0)  # must not raise
        self.assertEqual(effector.n_in_flight, 2)

        with self.assertRaises(TooManyInFlightException):
            effector.fire(self._target(), t0)

    def test_blocked_launch_does_not_consume_ammo_or_cadence(self):
        effector = self._make_effector(max_in_flight=1)
        effector.fire(self._target(), t0)
        n_attacks_before = effector.n_attacks_left
        time_of_last_shot_before = effector.time_of_last_shot

        with self.assertRaises(TooManyInFlightException):
            effector.fire(self._target(), t0)

        self.assertEqual(effector.n_attacks_left, n_attacks_before)
        self.assertEqual(effector.time_of_last_shot, time_of_last_shot_before)


if __name__ == "__main__":
    unittest.main()
