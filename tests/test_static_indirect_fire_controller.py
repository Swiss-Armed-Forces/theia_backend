import datetime
import unittest

import numpy as np

from theia.config import SIDC
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.effectors import DirectFireEffector
from theia.simulation.controllers.homing_effector import HomingSystem
from theia.simulation.controllers.static_indirect_fire_controller import (
    StaticIndirectFireController,
)
from theia.terrain import SrtmTerrainModel
from theia.types import (
    AbstractEventListener,
    ConstantRcsModel,
    Entity,
    IdProvider,
    KillEvent,
    Point,
    SituationalPicture,
    TextEvent,
    Track,
)

srtm = SrtmTerrainModel()

p_launcher = Point(
    lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
    lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
    alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
)

# Within 100 km of p_launcher (mirrors tests/test_static_direct_file_controller.py).
p_close = CoordinateTransformations.geodetic_to_cartesian(
    lat=47.37348,
    lon=8.53707,
    alt=1000.0,
)

# Far enough away (~3000 km) to be outside any reasonable launch distance.
p_far = CoordinateTransformations.geodetic_to_cartesian(
    lat=20.0,
    lon=8.53707,
    alt=1000.0,
)

t0 = datetime.datetime.fromtimestamp(0)
t1 = datetime.datetime.fromtimestamp(1)
dt = datetime.timedelta(seconds=1)


class RecordingListener(AbstractEventListener):
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def make_track(track_id: str, position: tuple[float, float, float]) -> Track:
    x, y, z = position
    return Track(
        id=track_id,
        sidc=SIDC.UNKNOWN,
        states=[
            (t0, np.array([x, 0.0, y, 0.0, z, 0.0])),
            (t1, np.array([x, 0.0, y, 0.0, z, 0.0])),
        ],
    )


def get_projectile() -> HomingSystem:
    effector = DirectFireEffector(
        id=99,
        point=p_launcher,
        combat_range=5_000,
        n_attacks_left=1,
        name="",
        terrain=srtm,
    )
    return HomingSystem(
        target_id=-1,
        sidc=SIDC.BLUE_MISSILE,
        speed=300.0,
        max_dist=50_000,
        point=p_launcher,
        rcs=ConstantRcsModel(rcs=0.1),
        effector=effector,
        assigned_track_id="-1",
        terrain=srtm,
    )


def get_controller(
    assigned_track_id: str | None = "0",
    n_shots_left: int = 1,
    id_provider: IdProvider | None = None,
    launch_distance: float = 100_000,
) -> StaticIndirectFireController:
    return StaticIndirectFireController(
        target_id=4,
        sidc=SIDC.BLUE_AIR_DEFENCE,
        rcs=1.5,
        projectile=get_projectile(),
        id_provier=id_provider if id_provider is not None else IdProvider(),
        n_shots_left=n_shots_left,
        launch_distance=launch_distance,
        assigned_track_id=assigned_track_id,
    )


def get_situational_picture(tracks: list[Track]) -> SituationalPicture:
    return SituationalPicture(
        time=t0,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=tracks,
    )


class LaunchConditionsTest(unittest.TestCase):
    def test_no_assigned_track_does_not_launch(self):
        controller = get_controller(assigned_track_id=None)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)

        self.assertEqual(len(controller._children), 0)

    def test_no_shots_left_does_not_launch(self):
        controller = get_controller(assigned_track_id="0", n_shots_left=0)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)

        self.assertEqual(len(controller._children), 0)

    def test_assigned_track_not_in_picture_does_not_launch(self):
        controller = get_controller(assigned_track_id="0")
        # Only an unrelated track is visible; the assigned one is absent.
        picture = get_situational_picture([make_track("1", p_close)])

        controller.update(picture, dt)

        self.assertEqual(len(controller._children), 0)

    def test_track_out_of_launch_range_does_not_launch(self):
        controller = get_controller(assigned_track_id="0", launch_distance=100_000)
        picture = get_situational_picture([make_track("0", p_far)])

        controller.update(picture, dt)

        self.assertEqual(len(controller._children), 0)

    def test_track_within_launch_range_launches(self):
        controller = get_controller(assigned_track_id="0", launch_distance=100_000)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)

        self.assertEqual(len(controller._children), 1)
        launched = controller._children[0].child
        self.assertIsInstance(launched, HomingSystem)
        self.assertEqual(launched.assigned_track_id, "0")

    def test_launch_consumes_one_shot(self):
        controller = get_controller(assigned_track_id="0", n_shots_left=3)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)

        self.assertEqual(controller.n_shots_left, 2)

    def test_no_launch_does_not_consume_ammunition(self):
        controller = get_controller(assigned_track_id=None, n_shots_left=3)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)

        self.assertEqual(controller.n_shots_left, 3)


class SingleLaunchPerUpdateTest(unittest.TestCase):
    def test_repeated_updates_launch_only_once_while_projectile_is_alive(self):
        controller = get_controller(assigned_track_id="0", n_shots_left=5)
        listener = RecordingListener()
        controller.register_event_listener(listener)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)
        controller.update(picture, dt)
        controller.update(picture, dt)

        self.assertEqual(len(controller._children), 1)
        self.assertEqual(len(listener.events), 1)
        self.assertEqual(controller.n_shots_left, 4)

    def test_new_projectile_launched_once_previous_one_is_dead(self):
        controller = get_controller(assigned_track_id="0", n_shots_left=2)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)
        self.assertEqual(len(controller._children), 1)
        first_child = controller._children[0]

        # A KillEvent addressed to the projectile arrives at the controller
        # (e.g. relayed by the Simulator) and must be forwarded down to it.
        controller.on_event(KillEvent(id=1, time=t0, target_id=first_child.target_id))
        controller.update(picture, dt)

        self.assertEqual(len(controller._children), 1)
        self.assertIsNot(controller._children[0], first_child)


class EventForwardingTest(unittest.TestCase):
    def test_on_event_forwards_to_children(self):
        controller = get_controller(assigned_track_id="0", n_shots_left=1)
        picture = get_situational_picture([make_track("0", p_close)])
        controller.update(picture, dt)
        child = controller._children[0]
        self.assertTrue(child._is_alive)

        controller.on_event(KillEvent(id=1, time=t0, target_id=child.target_id))

        self.assertFalse(child._is_alive)

    def test_on_event_forwards_to_a_newly_launched_child_too(self):
        controller = get_controller(assigned_track_id="0", n_shots_left=2)
        picture = get_situational_picture([make_track("0", p_close)])
        controller.update(picture, dt)
        first_child = controller._children[0]
        # Free up the track so a second, independent projectile can launch.
        controller.on_event(
            KillEvent(id=1, time=t0, target_id=first_child.target_id)
        )
        controller.update(picture, dt)
        second_child = controller._children[0]
        self.assertIsNot(first_child, second_child)

        controller.on_event(KillEvent(id=2, time=t0, target_id=second_child.target_id))

        self.assertFalse(second_child._is_alive)

    def test_unrelated_kill_event_does_not_kill_child(self):
        controller = get_controller(assigned_track_id="0", n_shots_left=1)
        picture = get_situational_picture([make_track("0", p_close)])
        controller.update(picture, dt)
        child = controller._children[0]

        controller.on_event(
            KillEvent(id=1, time=t0, target_id=child.target_id + 999)
        )

        self.assertTrue(child._is_alive)

    def test_on_event_with_no_children_does_not_raise(self):
        controller = get_controller(assigned_track_id=None)

        controller.on_event(KillEvent(id=1, time=t0, target_id=0))


class ProjectileIdTest(unittest.TestCase):
    def test_projectile_receives_freshly_minted_id(self):
        id_provider = IdProvider()
        # Reserve earlier IDs to prove the controller pulls a real
        # "next free" ID instead of e.g. hard-coding 0.
        id_provider.increment(Entity.TARGET)
        id_provider.increment(Entity.TARGET)
        controller = get_controller(assigned_track_id="0", id_provider=id_provider)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)

        launched = controller._children[0].child
        self.assertEqual(launched.target_id, 2)

    def test_successive_launches_get_distinct_ids(self):
        controller = get_controller(assigned_track_id="0", n_shots_left=2)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)
        first_id = controller._children[0].child.target_id
        controller.on_event(
            KillEvent(id=1, time=t0, target_id=controller._children[0].target_id)
        )
        controller.update(picture, dt)
        second_id = controller._children[0].child.target_id

        self.assertNotEqual(first_id, second_id)


class LaunchEventTest(unittest.TestCase):
    def test_launch_broadcasts_text_event_with_expected_message(self):
        controller = get_controller(assigned_track_id="0")
        listener = RecordingListener()
        controller.register_event_listener(listener)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)

        self.assertEqual(len(listener.events), 1)
        event = listener.events[0]
        self.assertIsInstance(event, TextEvent)
        self.assertEqual(event.time, picture.time + dt)
        launched_id = controller._children[0].child.target_id
        self.assertEqual(
            event.text,
            f"Launcher #4 ⤼ Track #0 (projectile #{launched_id})",
        )

    def test_no_launch_broadcasts_no_event(self):
        controller = get_controller(assigned_track_id=None)
        listener = RecordingListener()
        controller.register_event_listener(listener)
        picture = get_situational_picture([make_track("0", p_close)])

        controller.update(picture, dt)

        self.assertEqual(len(listener.events), 0)


if __name__ == "__main__":
    unittest.main()
