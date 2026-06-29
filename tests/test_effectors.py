import datetime
import unittest

import numpy as np

from theia.config import SIDC, UNKNOWN_ID, UNKNOWN_TIME
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.effectors import (
    DirectFireEffector,
    IndirectFireEffector,
    NoLosException,
    OutOfAttacksException,
    OutOfRangeException,
)
from theia.terrain import SrtmTerrainModel
from theia.types import (
    DirectShot,
    IndirectShot,
    Point,
    Track,
)

srtm = SrtmTerrainModel()

p_uetliberg = Point(
    lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
    lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
    alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
)

p_bern = CoordinateTransformations.geodetic_to_cartesian(
    lat=46.948056,
    lon=7.4475,
    alt=1000.0,
)


p_close = CoordinateTransformations.geodetic_to_cartesian(
    lat=47.37348,
    lon=8.53707,
    alt=1000.0,
)

p_no_los = CoordinateTransformations.geodetic_to_cartesian(
    lat=47.22726,
    lon=8.66719,
    alt=srtm.elevationAt(47.22726, 8.66719),
)


class DirectEffectorTest(unittest.TestCase):

    def test_too_far_away(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=10_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
        )
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_bern, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        with self.assertRaises(OutOfRangeException):
            effector.fire(track)

    def test_no_attacks_left(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=10_000,
            n_attacks_left=0,
            name="",
            terrain=srtm,
        )
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_close, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        with self.assertRaises(OutOfAttacksException):
            effector.fire(track)

    def test_no_los(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
        )
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_no_los, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        with self.assertRaises(NoLosException):
            effector.fire(track)

    def test_use_ammo(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
        )
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_close, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        self.assertEqual(effector.n_attacks_left, 1)
        effector.fire(track)
        self.assertEqual(effector.n_attacks_left, 0)

        with self.assertRaises(OutOfAttacksException):
            effector.fire(track)

    def test_result(self):
        effector = DirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            terrain=srtm,
        )
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_close, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        event = effector.fire(track)
        self.assertIsInstance(event, DirectShot)
        self.assertEqual(event.track, track)
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
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_bern, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        with self.assertRaises(OutOfRangeException):
            effector.fire(track)

    def test_no_attacks_left(self):
        effector = IndirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=10_000,
            n_attacks_left=0,
            name="",
            projectile=self._projectile,
        )
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_close, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        with self.assertRaises(OutOfAttacksException):
            effector.fire(track)

    def test_los_irrelevant(self):
        effector = IndirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            projectile=self._projectile,
        )
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_no_los, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        effector.fire(track)

    def test_use_ammo(self):
        effector = IndirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            projectile=self._projectile,
        )
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_close, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        self.assertEqual(effector.n_attacks_left, 1)
        effector.fire(track)
        self.assertEqual(effector.n_attacks_left, 0)

        with self.assertRaises(OutOfAttacksException):
            effector.fire(track)

    def test_event(self):
        effector = IndirectFireEffector(
            id=0,
            point=p_uetliberg,
            combat_range=100_000,
            n_attacks_left=1,
            name="",
            projectile=self._projectile,
        )
        track = Track(
            id="0",
            sidc=SIDC.UNKNOWN,
            states=[
                (
                    datetime.datetime.fromtimestamp(0),
                    np.array([*p_close, 0.0, 0.0, 0.0]),
                ),
            ],
        )

        event = effector.fire(track)
        self.assertIsInstance(event, IndirectShot)
        self.assertEqual(event.track, track)
        self.assertEqual(event.shooter, effector)
        self.assertEqual(event.id, UNKNOWN_ID)
        self.assertEqual(event.time, UNKNOWN_TIME)
        self.assertEqual(event.projectile, self._projectile)


if __name__ == "__main__":
    unittest.main()
