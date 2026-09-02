import datetime
import unittest

import numpy as np

from theia.config import SIDC
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.effectors import DirectFireEffector
from theia.simulation.controllers.static_direct_fire_controller import (
    StaticDirectFireController,
)
from theia.terrain import SrtmTerrainModel
from theia.types import (
    ConstantRcsModel,
    Point,
    SituationalPicture,
    Target,
    Track,
    Velocity,
)

srtm = SrtmTerrainModel()

p_uetliberg = Point(
    lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
    lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
    alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
)

p_close = CoordinateTransformations.geodetic_to_cartesian(
    lat=47.37348,
    lon=8.53707,
    alt=1000.0,
)

p_close2 = CoordinateTransformations.geodetic_to_cartesian(
    lat=47.37790,
    lon=8.49466,
    alt=1000.0,
)

t0 = datetime.datetime.fromtimestamp(0)
t1 = datetime.datetime.fromtimestamp(1)


def get_effector() -> DirectFireEffector:
    return DirectFireEffector(
        id=5,
        point=p_uetliberg,
        combat_range=100_000,
        n_attacks_left=1,
        name="",
        terrain=srtm,
        # Cadence is checked/tracked entirely by the effector's own fire()
        # (see tests/test_effectors.py::CadenceTest) - unlimited here since
        # these tests are about the controller's targeting decision, not
        # cadence.
        cadence=float("inf"),
    )


def get_controller(assigned_track_id: str | None) -> StaticDirectFireController:
    return StaticDirectFireController(
        target_id=4,
        sidc=SIDC.BLUE_AIR_DEFENCE,
        rcs=1.5,
        effector=get_effector(),
        assigned_track_id=assigned_track_id,
    )


def get_situational_picture_in_range() -> SituationalPicture:
    track = Track(
        id="0",
        sidc=SIDC.UNKNOWN,
        states=[
            (
                t0,
                np.array([p_close[0], 0.0, p_close[1], 0.0, p_close[2], 0.0]),
            ),
            (
                t1,
                np.array([p_close[0], 0.0, p_close[1], 0.0, p_close[2], 0.0]),
            ),
        ],
    )
    # Used to check whether the controller just takes the first track or actually searches.
    red_herring = Track(
        id="1",
        sidc=SIDC.UNKNOWN,
        states=[
            (
                t0,
                np.array([p_close2[0], 0.0, p_close2[1], 0.0, p_close2[2], 0.0]),
            ),
            (
                t1,
                np.array([p_close2[0], 0.0, p_close2[1], 0.0, p_close2[2], 0.0]),
            ),
        ],
    )
    return SituationalPicture(
        time=t0,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[red_herring, track],
    )


class StaticDirectFireControllerTest(unittest.TestCase):
    def test_no_assigned_target(self):
        controller = get_controller(None)
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))
        fire_decisions = controller.firing_effectors
        self.assertEqual(len(fire_decisions), 0)

    def test_assigned_target(self):
        controller = get_controller("0")
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))

        fire_decisions = controller.firing_effectors
        self.assertEqual(len(fire_decisions), 1)

        effector, p = fire_decisions[0]

        self.assertEqual(effector, controller.effector)
        self.assertAlmostEqual(p.lat, 47.37348)
        self.assertAlmostEqual(p.lon, 8.53707)
        self.assertAlmostEqual(p.alt, 1000.0)

    def test_as_target(self):
        controller = get_controller(None)
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))
        targets = controller.targets

        expected = Target(
            id=4,
            is_stationary=True,
            sidc=SIDC.BLUE_AIR_DEFENCE,
            point=p_uetliberg,
            cross_section_model=ConstantRcsModel(rcs=1.5),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], expected)


if __name__ == "__main__":
    unittest.main()
