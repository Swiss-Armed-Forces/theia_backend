import datetime
import unittest
from unittest.mock import patch

from theia.simulation.controllers.living_controller import (
    AlreadyDeadException,
    LivingController,
)
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.simulator import KillEvent
from theia.test_data import get_uetliberg_radar
from theia.types import ConstantRcsModel, Event, SituationalPicture


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
        with patch.object(self._radar_controller, "get_monostatic_radars") as mock:
            self._controller.get_monostatic_radars(picture, dt)
            mock.assert_called_once()

        with patch.object(self._radar_controller, "get_pcl_sensors") as mock:
            self._controller.get_pcl_sensors(picture, dt)
            mock.assert_called_once()

        with patch.object(self._radar_controller, "get_targets") as mock:
            self._controller.get_targets(picture, dt)
            mock.assert_called_once()

        with patch.object(self._radar_controller, "get_pet_receivers") as mock:
            self._controller.get_pet_receivers(picture, dt)
            mock.assert_called_once()

        with patch.object(self._radar_controller, "get_firing_effectors") as mock:
            self._controller.get_firing_effectors(picture, dt)
            mock.assert_called_once()

    def test_kill(self):
        self.assertTrue(self._controller._is_alive)
        self._controller.on_event(Event(id=1, time=get_t0()))
        self.assertTrue(self._controller._is_alive)
        self._controller.on_event(KillEvent(id=2, time=get_t0(), target_id=0))
        self.assertTrue(self._controller._is_alive)
        self._controller.on_event(KillEvent(id=3, time=get_t0(), target_id=2))
        self.assertFalse(self._controller._is_alive)

    def test_double_kill_raises(self):
        self.assertTrue(self._controller._is_alive)
        self._controller.on_event(KillEvent(id=3, time=get_t0(), target_id=2))
        with self.assertRaises(AlreadyDeadException):
            self._controller.on_event(KillEvent(id=4, time=get_t0(), target_id=2))

    def test_dead(self):
        self.assertTrue(self._controller._is_alive)
        self._controller.on_event(KillEvent(id=3, time=get_t0(), target_id=2))
        self.assertFalse(self._controller._is_alive)
        picture = get_dummy_situational_picture()
        dt = get_dt()
        with patch.object(self._radar_controller, "get_monostatic_radars") as mock:
            result = self._controller.get_monostatic_radars(picture, dt)
            mock.assert_not_called()
            self.assertEqual(len(result), 0)

        with patch.object(self._radar_controller, "get_pcl_sensors") as mock:
            result = self._controller.get_pcl_sensors(picture, dt)
            mock.assert_not_called()
            self.assertEqual(len(result), 0)

        with patch.object(self._radar_controller, "get_targets") as mock:
            result = self._controller.get_targets(picture, dt)
            mock.assert_not_called()
            self.assertEqual(len(result), 0)

        with patch.object(self._radar_controller, "get_pet_receivers") as mock:
            result = self._controller.get_pet_receivers(picture, dt)
            mock.assert_not_called()
            self.assertEqual(len(result), 0)

        with patch.object(self._radar_controller, "get_firing_effectors") as mock:
            result = self._controller.get_firing_effectors(picture, dt)
            mock.assert_not_called()
            self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
