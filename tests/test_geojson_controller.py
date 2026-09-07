import datetime
import unittest
from unittest.mock import MagicMock, patch

from theia.simulation.controllers.geojson_controller import GeoJsonController
from theia.types import (
    Controller,
    Event,
    GeoJSONFeature,
    GeoJSONPolygon,
    SituationalPicture,
)


class FakeChildController(Controller):
    """Minimal concrete Controller usable as a child in tests."""

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        pass

    def on_event(self, event: Event):
        pass

    def add_controller(self, controller: Controller):
        pass


def get_dummy_situational_picture() -> SituationalPicture:
    return SituationalPicture(
        time=datetime.datetime.fromtimestamp(0, datetime.UTC),
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


class GeoJsonControllerTest(unittest.TestCase):
    def setUp(self):
        self._child = FakeChildController()
        self._controller = GeoJsonController(child=self._child, geojson_features={})

    def test_events_broadcast_by_child_reach_listeners_registered_on_wrapper(self):
        # Regression test: events raised deep inside the wrapped controller
        # tree (e.g. a HomingSystem committing suicide) must bubble up
        # through GeoJsonController to whoever listens on it (the
        # Simulator). Previously GeoJsonController never relayed the
        # child's own broadcasts, silently swallowing them, which meant
        # self-inflicted KillEvents never made it back down to the
        # corresponding LivingController and the entity was never removed.
        listener = MagicMock()
        self._controller.register_event_listener(listener)

        event = Event(id=1, time=datetime.datetime.fromtimestamp(0, datetime.UTC))
        self._child._broadcast_event(event)

        listener.on_event.assert_called_once_with(event)

    def test_on_event_forwards_to_child(self):
        event = Event(id=1, time=datetime.datetime.fromtimestamp(0, datetime.UTC))
        with patch.object(
            FakeChildController, "on_event", autospec=True
        ) as mock_on_event:
            self._controller.on_event(event)
            mock_on_event.assert_called_once_with(self._child, event)

    def test_add_controller_forwards_to_child(self):
        grandchild = FakeChildController()
        with patch.object(
            FakeChildController, "add_controller", autospec=True
        ) as mock_add:
            self._controller.add_controller(grandchild)
            mock_add.assert_called_once_with(self._child, grandchild)

    def test_update_merges_child_state_and_own_geojson_features(self):
        own_feature = GeoJSONFeature(
            geometry=GeoJSONPolygon(coordinates=[[[0.0, 0.0]]])
        )
        child_feature = GeoJSONFeature(
            geometry=GeoJSONPolygon(coordinates=[[[1.0, 1.0]]])
        )
        self._controller.geojson_features = {"own": own_feature}
        self._child.geojson = {"child": child_feature}

        picture = get_dummy_situational_picture()
        dt = datetime.timedelta(seconds=1)
        with patch.object(FakeChildController, "update", autospec=True) as mock_update:
            self._controller.update(picture, dt)
            mock_update.assert_called_once_with(self._child, picture, dt)

        self.assertEqual(
            self._controller.geojson,
            {"child": child_feature, "own": own_feature},
        )


if __name__ == "__main__":
    unittest.main()
