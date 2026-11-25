import datetime
import unittest

from theia.config import UNKNOWN_RCS
from theia.simulation.controllers.intel_controller import IntelController
from theia.types import (
    ConstantRcsModel,
    IntelSensor,
    SituationalPicture,
    TargetInfos,
    TextEvent,
    Trajectory,
)

t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
dt = datetime.timedelta(seconds=1)


def get_intel_sensor() -> IntelSensor:
    return IntelSensor(
        id=3,
        trajectory=Trajectory(
            target_id=4,
            target_info=TargetInfos.UNKNOWN,
            times=[t0, t0 + datetime.timedelta(seconds=10)],
            lats=[47.0] * 2,
            lons=[8.0] * 2,
            alts=[500.0] * 2,
            cross_section_model=ConstantRcsModel(rcs=UNKNOWN_RCS),
        ),
    )


def get_picture(time: datetime.datetime = t0) -> SituationalPicture:
    return SituationalPicture(
        time=time,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


class IntelControllerTest(unittest.TestCase):
    def assert_other_outputs_empty(self, controller: IntelController):
        self.assertEqual(controller.monostatic_sensors, [])
        self.assertEqual(controller.pcl_sensors, [])
        self.assertEqual(controller.targets, [])
        self.assertEqual(controller.pet_receivers, [])
        self.assertEqual(controller.visual_sensors, [])
        self.assertEqual(controller.firing_effectors, [])
        self.assertEqual(controller.geojson, [])

    def test_exposes_its_sensor_only(self):
        sensor = get_intel_sensor()
        controller = IntelController(sensor=sensor)
        controller.update(get_picture(t0), dt)

        self.assertEqual(len(controller.intel_sensors), 1)
        self.assertIs(controller.intel_sensors[0], sensor)
        self.assert_other_outputs_empty(controller)

    def test_output_is_stable_over_ticks(self):
        sensor = get_intel_sensor()
        controller = IntelController(sensor=sensor)
        for i in range(3):
            with self.subTest(tick=i):
                controller.update(get_picture(t0 + i * dt), dt)
                self.assertEqual(len(controller.intel_sensors), 1)
                self.assertIs(controller.intel_sensors[0], sensor)
                self.assert_other_outputs_empty(controller)

    def test_ignores_events(self):
        sensor = get_intel_sensor()
        controller = IntelController(sensor=sensor)
        controller.update(get_picture(t0), dt)
        controller.on_event(TextEvent(id=1, time=t0, text="ignored"))

        self.assertEqual(len(controller.intel_sensors), 1)
        self.assertIs(controller.intel_sensors[0], sensor)
        self.assert_other_outputs_empty(controller)


if __name__ == "__main__":
    unittest.main()
