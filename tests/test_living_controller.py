import datetime
import unittest
from unittest.mock import patch

from theia.simulation.controllers.living_controller import (
    LivingController,
)
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.simulator import KillEvent
from theia.test_data import get_uetliberg_radar
from theia.types import (
    ConstantRcsModel,
    Event,
    GeoJSONAnnotation,
    GeoJSONFeature,
    GeoJSONPolygon,
    SituationalPicture,
)


def get_t0() -> datetime.datetime:
    return datetime.datetime.fromtimestamp(0)


def get_dt() -> datetime.timedelta:
    return datetime.timedelta(seconds=1)


def get_dummy_situational_picture() -> SituationalPicture:
    return SituationalPicture(
        time=get_t0(),
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


class LivingControllerTest(unittest.TestCase):
    def setUp(self):
        TARGET_ID = 2
        self._radar = get_uetliberg_radar()
        self._radar_controller = MonostaticRadarController(
            target_id=TARGET_ID,
            radar=self._radar,
            is_blue=True,
            rcs_model=ConstantRcsModel(rcs=1.0),
            name="my radar",
        )
        self._controller = LivingController(
            child=self._radar_controller,
            target_id=TARGET_ID,
        )

    def test_alive(self):
        self.assertTrue(self._controller._is_alive)
        picture = get_dummy_situational_picture()
        dt = get_dt()
        with patch.object(MonostaticRadarController, "update", autospec=True) as mock:
            self._controller.update(picture, dt)
            mock.assert_called_once_with(self._radar_controller, picture, dt)

    def test_kill(self):
        self.assertTrue(self._controller._is_alive)
        self._controller.on_event(Event(id=1, time=get_t0()))
        self.assertTrue(self._controller._is_alive)
        self._controller.on_event(KillEvent(id=2, time=get_t0(), target_id=0))
        self.assertTrue(self._controller._is_alive)
        self._controller.on_event(KillEvent(id=3, time=get_t0(), target_id=2))
        self.assertFalse(self._controller._is_alive)

    def test_is_alive_property(self):
        self.assertTrue(self._controller.is_alive)
        self._controller.on_event(KillEvent(id=3, time=get_t0(), target_id=2))
        self.assertFalse(self._controller.is_alive)

    def test_dead(self):
        self.assertTrue(self._controller._is_alive)
        self._controller.on_event(KillEvent(id=3, time=get_t0(), target_id=2))
        self.assertFalse(self._controller._is_alive)
        picture = get_dummy_situational_picture()
        dt = get_dt()
        with patch.object(MonostaticRadarController, "update", autospec=True) as mock:
            self._controller.update(picture, dt)
            mock.assert_not_called()
            self.assertEqual(len(self._controller.monostatic_sensors), 0)
            self.assertEqual(len(self._controller.pcl_sensors), 0)
            self.assertEqual(len(self._controller.targets), 0)
            self.assertEqual(len(self._controller.pet_receivers), 0)
            self.assertEqual(len(self._controller.firing_effectors), 0)

    def test_geojson_passed_through_while_alive_and_hidden_when_dead(self):
        annotation = GeoJSONAnnotation(
            id="range",
            feature=GeoJSONFeature(geometry=GeoJSONPolygon(coordinates=[[[0.0, 0.0]]])),
        )
        self._radar_controller.geojson = [annotation]
        picture = get_dummy_situational_picture()
        with patch.object(MonostaticRadarController, "update", autospec=True):
            self._controller.update(picture, get_dt())
            self.assertEqual(self._controller.geojson, [annotation])

            self._controller.on_event(KillEvent(id=3, time=get_t0(), target_id=2))
            self._controller.update(picture, get_dt())
            self.assertEqual(self._controller.geojson, [])


if __name__ == "__main__":
    unittest.main()
