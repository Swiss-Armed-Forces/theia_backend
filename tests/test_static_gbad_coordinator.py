import datetime
import unittest

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.effectors import DirectFireEffector
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.static_gbad_controller import StaticGbadController
from theia.simulation.controllers.static_gbad_coordinator import StaticGbadCoordinator
from theia.terrain import AbstractTerrainModel
from theia.types import (
    KillEvent,
    Party,
    Point,
    SituationalPicture,
    TargetInfos,
    Track,
)

t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
t1 = datetime.datetime.fromtimestamp(1, datetime.UTC)
dt = datetime.timedelta(seconds=1)

P_NEAR = Point(lat=47.0, lon=8.0, alt=500.0)
"""Effector position ~1.2 km from ``P_TRACK``."""
P_FAR = Point(lat=47.0, lon=8.05, alt=500.0)
"""Effector position ~3.0 km from ``P_TRACK``."""
P_TRACK = Point(lat=47.0, lon=8.01, alt=1000.0)
P_OTHER_TRACK = Point(lat=47.0, lon=8.06, alt=1000.0)
"""Track position closer to ``P_FAR`` than to ``P_NEAR``."""


class FlatTerrain(AbstractTerrainModel):
    """
    Flat terrain with line of sight everywhere, except from/to the effector
    positions listed in ``blocked``.
    """

    step_m: float = 30.0
    blocked: list[Point] = []

    def elevationAt(self, lat, lon):
        return 0.0

    def has_line_of_sight(self, p1: Point, p2: Point) -> bool:
        return p1 not in self.blocked and p2 not in self.blocked


def get_track(track_id: str, p: Point) -> Track:
    x, y, z = CoordinateTransformations.geodetic_to_cartesian(p.lat, p.lon, p.alt)
    state = np.array([x, 0.0, y, 0.0, z, 0.0])
    return Track(
        id=track_id,
        target_info=TargetInfos.UNKNOWN,
        states=[(t0, state), (t1, state)],
    )


def get_picture(tracks: list[Track]) -> SituationalPicture:
    return SituationalPicture(
        time=t0,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=tracks,
    )


def get_controller(
    target_id: int,
    p: Point,
    terrain: AbstractTerrainModel,
    combat_range: float = 100_000,
) -> LivingController[StaticGbadController]:
    effector = DirectFireEffector(
        id=target_id,
        point=p,
        combat_range=combat_range,
        n_attacks_left=10,
        name="",
        terrain=terrain,
        cadence=float("inf"),
    )
    return LivingController(
        child=StaticGbadController(
            target_id=target_id,
            info=TargetInfos.GBAD.with_party(Party.BLUE),
            rcs=1.5,
            effector=effector,
        ),
        target_id=target_id,
    )


def assigned(controller: LivingController[StaticGbadController]) -> str | None:
    return controller.child.assigned_track_id


class StaticGbadCoordinatorTest(unittest.TestCase):
    def test_no_tracks_no_assignment(self):
        terrain = FlatTerrain()
        c = get_controller(1, P_NEAR, terrain)
        coordinator = StaticGbadCoordinator([c], terrain)

        coordinator.update(get_picture([]), dt)

        self.assertIsNone(assigned(c))
        self.assertEqual(list(coordinator.firing_effectors), [])

    def test_single_effector_in_range_is_assigned_and_fires(self):
        terrain = FlatTerrain()
        c = get_controller(1, P_NEAR, terrain)
        coordinator = StaticGbadCoordinator([c], terrain)

        coordinator.update(get_picture([get_track("a", P_TRACK)]), dt)

        self.assertEqual(assigned(c), "a")
        firing = list(coordinator.firing_effectors)
        self.assertEqual(len(firing), 1)
        effector, p = firing[0]
        self.assertIs(effector, c.child.effector)
        self.assertAlmostEqual(p.lat, P_TRACK.lat)
        self.assertAlmostEqual(p.lon, P_TRACK.lon)
        self.assertAlmostEqual(p.alt, P_TRACK.alt)

    def test_closest_effector_is_assigned(self):
        # The closest controller is deliberately not the last one in the list.
        terrain = FlatTerrain()
        near = get_controller(1, P_NEAR, terrain)
        far = get_controller(2, P_FAR, terrain)
        coordinator = StaticGbadCoordinator([near, far], terrain)

        coordinator.update(get_picture([get_track("a", P_TRACK)]), dt)

        self.assertEqual(assigned(near), "a")
        self.assertIsNone(assigned(far))

    def test_closest_effector_is_assigned_regardless_of_order(self):
        terrain = FlatTerrain()
        near = get_controller(1, P_NEAR, terrain)
        far = get_controller(2, P_FAR, terrain)
        coordinator = StaticGbadCoordinator([far, near], terrain)

        coordinator.update(get_picture([get_track("a", P_TRACK)]), dt)

        self.assertEqual(assigned(near), "a")
        self.assertIsNone(assigned(far))

    def test_out_of_range_effector_is_not_assigned(self):
        terrain = FlatTerrain()
        c = get_controller(1, P_NEAR, terrain, combat_range=500)
        coordinator = StaticGbadCoordinator([c], terrain)

        coordinator.update(get_picture([get_track("a", P_TRACK)]), dt)

        self.assertIsNone(assigned(c))
        self.assertEqual(list(coordinator.firing_effectors), [])

    def test_out_of_range_closest_falls_back_to_next_in_range(self):
        terrain = FlatTerrain()
        near = get_controller(1, P_NEAR, terrain, combat_range=500)
        far = get_controller(2, P_FAR, terrain)
        coordinator = StaticGbadCoordinator([near, far], terrain)

        coordinator.update(get_picture([get_track("a", P_TRACK)]), dt)

        self.assertIsNone(assigned(near))
        self.assertEqual(assigned(far), "a")

    def test_no_line_of_sight_falls_back_to_next_closest(self):
        terrain = FlatTerrain(blocked=[P_NEAR])
        near = get_controller(1, P_NEAR, terrain)
        far = get_controller(2, P_FAR, terrain)
        coordinator = StaticGbadCoordinator([far, near], terrain)

        coordinator.update(get_picture([get_track("a", P_TRACK)]), dt)

        self.assertIsNone(assigned(near))
        self.assertEqual(assigned(far), "a")

    def test_no_line_of_sight_at_all_no_assignment(self):
        terrain = FlatTerrain(blocked=[P_NEAR])
        c = get_controller(1, P_NEAR, terrain)
        coordinator = StaticGbadCoordinator([c], terrain)

        coordinator.update(get_picture([get_track("a", P_TRACK)]), dt)

        self.assertIsNone(assigned(c))

    def test_tracks_are_distributed_to_their_closest_effectors(self):
        terrain = FlatTerrain()
        near = get_controller(1, P_NEAR, terrain)
        far = get_controller(2, P_FAR, terrain)
        coordinator = StaticGbadCoordinator([near, far], terrain)

        coordinator.update(
            get_picture([get_track("a", P_TRACK), get_track("b", P_OTHER_TRACK)]),
            dt,
        )

        self.assertEqual(assigned(near), "a")
        self.assertEqual(assigned(far), "b")
        self.assertEqual(len(list(coordinator.firing_effectors)), 2)

    def test_previous_assignment_is_reset(self):
        terrain = FlatTerrain()
        c = get_controller(1, P_NEAR, terrain)
        coordinator = StaticGbadCoordinator([c], terrain)

        coordinator.update(get_picture([get_track("a", P_TRACK)]), dt)
        self.assertEqual(assigned(c), "a")

        coordinator.update(get_picture([]), dt)
        self.assertIsNone(assigned(c))
        self.assertEqual(list(coordinator.firing_effectors), [])

    def test_killed_effector_is_not_assigned(self):
        terrain = FlatTerrain()
        near = get_controller(1, P_NEAR, terrain)
        far = get_controller(2, P_FAR, terrain)
        coordinator = StaticGbadCoordinator([near, far], terrain)

        coordinator.on_event(KillEvent(id=0, time=t0, target_id=1))
        coordinator.update(get_picture([get_track("a", P_TRACK)]), dt)

        self.assertEqual(assigned(far), "a")
        firing = list(coordinator.firing_effectors)
        self.assertEqual(len(firing), 1)
        self.assertIs(firing[0][0], far.child.effector)


if __name__ == "__main__":
    unittest.main()
