import datetime
import unittest

from theia.simulation.controllers.controller_group import ControllerGroup
from theia.types import AbstractEventListener, Controller, Event, SituationalPicture


def get_situational_picture() -> SituationalPicture:
    return SituationalPicture(
        time=datetime.datetime.fromtimestamp(0),
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


class StubController(Controller):
    updated: bool = False

    def on_event(self, event):
        pass

    def update(self, situational_picture, dt):
        self.updated = True


class DeadStubController(StubController):
    def model_post_init(self, context):
        super().model_post_init(context)
        self._is_alive = False


class RecordingListener(AbstractEventListener):
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


class ControllerGroupTest(unittest.TestCase):
    def test_add_controller_receives_updates(self):
        group = ControllerGroup([])
        stub = StubController()

        group.add_controller(stub)
        group.update(get_situational_picture(), datetime.timedelta(seconds=1))

        self.assertIn(stub, group._controllers)
        self.assertTrue(stub.updated)

    def test_add_controller_wires_event_relais(self):
        group = ControllerGroup([])
        stub = StubController()
        listener = RecordingListener()
        group.register_event_listener(listener)

        group.add_controller(stub)
        event = Event(id=1, time=datetime.datetime.fromtimestamp(0))
        stub._broadcast_event(event)

        self.assertEqual(listener.events, [event])

    def test_dead_controllers_are_pruned(self):
        dead = DeadStubController()
        alive = StubController()
        group = ControllerGroup([])
        group.add_controller(dead)
        group.add_controller(alive)

        group.update(get_situational_picture(), datetime.timedelta(seconds=1))

        self.assertNotIn(dead, group._controllers)
        self.assertIn(alive, group._controllers)

    def test_controller_without_is_alive_attribute_is_never_pruned(self):
        stub = StubController()
        group = ControllerGroup([])
        group.add_controller(stub)

        group.update(get_situational_picture(), datetime.timedelta(seconds=1))
        group.update(get_situational_picture(), datetime.timedelta(seconds=1))

        self.assertIn(stub, group._controllers)


if __name__ == "__main__":
    unittest.main()
