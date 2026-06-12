import unittest

from theia.config import SIDC, UNKNOWN_ID, UNKNOWN_TIME
from theia.coordinates import POSITIONS_OF_INTEREST
from theia.effectors import DirectFireEffector, IndirectFireEffector
from theia.terrain import SrtmTerrainModel
from theia.types import (
    AbstractEventListener,
    ConstantRcsModel,
    DirectFireEvent,
    Event,
    IndirectFireEvent,
    Point,
    Target,
    Velocity,
)

srtm = SrtmTerrainModel()

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


class DirectEffectorTest(unittest.TestCase, AbstractEventListener):
    def setUp(self):
        self._events: list[DirectFireEvent] = []

    def test_too_far_away(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=10_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_bern,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(ValueError):
            effector.fire(target)

    def test_no_attacks_left(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=10_000,
            n_attacks_left=0,
            name="",
            terrain=srtm,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(ValueError):
            effector.fire(target)

    def test_no_los(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_no_los,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(ValueError):
            effector.fire(target)

    def test_use_ammo(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        self.assertEqual(effector.n_attacks_left, 1)
        effector.fire(target)
        self.assertEqual(effector.n_attacks_left, 0)

        with self.assertRaises(ValueError):
            effector.fire(target)

    def test_event(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )
        effector.register_event_listener(self)

        self.assertEqual(self._events, [])
        effector.fire(target)
        self.assertEqual(len(self._events), 1)
        event = self._events[0]
        self.assertEqual(event.target, target)
        self.assertEqual(event.shooter, effector)
        self.assertEqual(event.id, UNKNOWN_ID)
        self.assertEqual(event.time, UNKNOWN_TIME)

    def on_event(self, event: Event):
        self._events.append(event)


class IndirectEffectorTest(unittest.TestCase, AbstractEventListener):
    def setUp(self):
        self._events: list[IndirectFireEvent] = []
        self._projectile = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
        )

    def test_too_far_away(self):
        effector = IndirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=10_000,
            n_attacks_left=1,
            name="",
            projectile=self._projectile,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_bern,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(ValueError):
            effector.fire(target)

    def test_no_attacks_left(self):
        effector = IndirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=10_000,
            n_attacks_left=0,
            name="",
            projectile=self._projectile,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        with self.assertRaises(ValueError):
            effector.fire(target)

    def test_los_irrelevant(self):
        effector = IndirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            projectile=self._projectile,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_no_los,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        effector.fire(target)

    def test_use_ammo(self):
        effector = IndirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            projectile=self._projectile,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        self.assertEqual(effector.n_attacks_left, 1)
        effector.fire(target)
        self.assertEqual(effector.n_attacks_left, 0)

        with self.assertRaises(ValueError):
            effector.fire(target)

    def test_event(self):
        effector = IndirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            projectile=self._projectile,
        )
        target = Target(
            id=0,
            is_stationary=False,
            sidc=SIDC.UNKNOWN,
            point=p_close,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )
        effector.register_event_listener(self)

        self.assertEqual(self._events, [])
        effector.fire(target)
        self.assertEqual(len(self._events), 1)
        event = self._events[0]
        self.assertEqual(event.target, target)
        self.assertEqual(event.shooter, effector)
        self.assertEqual(event.id, UNKNOWN_ID)
        self.assertEqual(event.time, UNKNOWN_TIME)
        self.assertEqual(event.projectile, self._projectile)

    def on_event(self, event: Event):
        self._events.append(event)


if __name__ == "__main__":
    unittest.main()
