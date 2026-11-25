import datetime
import unittest

from theia.config import UNKNOWN_RCS
from theia.simulation.controllers.known_target_controller import (
    KnownTargetController,
)
from theia.types import (
    Entity,
    IdProvider,
    Party,
    Point,
    SimulationContext,
    SituationalPicture,
    TargetInfos,
    TextEvent,
    TheiaException,
)

t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
dt = datetime.timedelta(seconds=1)
POINT = Point(lat=47.366, lon=7.967, alt=380.0)
INFO = TargetInfos.CRITICAL_INFRASTRUCTURE.with_party(Party.BLUE)


def get_picture(time: datetime.datetime = t0) -> SituationalPicture:
    return SituationalPicture(
        time=time,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


def get_controller(id_provider: IdProvider | None = None) -> KnownTargetController:
    controller = KnownTargetController(point=POINT, target_info=INFO)
    controller.bind(
        SimulationContext(
            id_provider=id_provider if id_provider is not None else IdProvider(),
            start_time=t0,
        )
    )
    return controller


class KnownTargetControllerTest(unittest.TestCase):
    def test_update_requires_binding(self):
        controller = KnownTargetController(point=POINT, target_info=INFO)
        with self.assertRaises(TheiaException):
            controller.update(get_picture(), dt)

    def test_exposes_one_intel_sensor_only(self):
        controller = get_controller()
        controller.update(get_picture(), dt)
        self.assertEqual(len(controller.intel_sensors), 1)
        self.assertEqual(controller.monostatic_sensors, [])
        self.assertEqual(controller.pcl_sensors, [])
        self.assertEqual(controller.targets, [])
        self.assertEqual(controller.pet_receivers, [])
        self.assertEqual(controller.visual_sensors, [])
        self.assertEqual(controller.firing_effectors, [])
        self.assertEqual(controller.geojson, [])

    def test_reports_point_and_target_info(self):
        controller = get_controller()
        controller.update(get_picture(), dt)
        [sensor] = controller.intel_sensors
        target = sensor.detect(t0)
        self.assertAlmostEqual(target.point.lat, POINT.lat)
        self.assertAlmostEqual(target.point.lon, POINT.lon)
        self.assertAlmostEqual(target.point.alt, POINT.alt)
        self.assertEqual(target.info, INFO)
        self.assertEqual(sensor.trajectory.cross_section_model.rcs, UNKNOWN_RCS)

    def test_target_info_is_used_as_given(self):
        # The party is the owner's belief, not overridden by the side.
        info = TargetInfos.AIRPORT.with_party(Party.NEUTRAL)
        controller = KnownTargetController(point=POINT, target_info=info)
        controller.bind(SimulationContext(id_provider=IdProvider(), start_time=t0))
        controller.update(get_picture(), dt)
        [sensor] = controller.intel_sensors
        self.assertEqual(sensor.trajectory.target_info, info)

    def test_is_unlimited_in_time(self):
        controller = get_controller()
        years = datetime.timedelta(days=3650)
        for time in (t0 - years, t0, t0 + years):
            with self.subTest(time=time):
                controller.update(get_picture(time), dt)
                [sensor] = controller.intel_sensors
                self.assertIsNotNone(sensor.detect(time))

    def test_ids_come_from_id_provider(self):
        id_provider = IdProvider()
        id_provider.register_entity(Entity.TARGET, 7)
        id_provider.register_entity(Entity.SENSOR, 3)
        controller = get_controller(id_provider)
        controller.update(get_picture(), dt)
        [sensor] = controller.intel_sensors
        self.assertEqual(sensor.trajectory.target_id, 8)
        self.assertEqual(sensor.id, 4)

    def test_controllers_get_distinct_ids(self):
        id_provider = IdProvider()
        sensors = []
        for _ in range(2):
            controller = get_controller(id_provider)
            controller.update(get_picture(), dt)
            sensors += controller.intel_sensors
        self.assertNotEqual(sensors[0].id, sensors[1].id)
        self.assertNotEqual(
            sensors[0].trajectory.target_id, sensors[1].trajectory.target_id
        )

    def test_rebinding_keeps_ids(self):
        id_provider = IdProvider()
        controller = get_controller(id_provider)
        controller.update(get_picture(), dt)
        [before] = controller.intel_sensors
        controller.bind(SimulationContext(id_provider=id_provider, start_time=t0))
        controller.update(get_picture(), dt)
        [after] = controller.intel_sensors
        self.assertEqual(before.id, after.id)
        self.assertEqual(before.trajectory.target_id, after.trajectory.target_id)

    def test_ignores_events_and_is_stable_over_updates(self):
        controller = get_controller()
        controller.update(get_picture(), dt)
        [before] = controller.intel_sensors
        controller.on_event(TextEvent(id=1, time=t0, text="hello"))
        controller.update(get_picture(t0 + dt), dt)
        self.assertEqual(controller.intel_sensors, [before])


if __name__ == "__main__":
    unittest.main()
