import datetime
import unittest

from theia.config import SIDC
from theia.simulation.controllers.critical_infrastructure_controller import (
    CriticalInfrastructureController,
)
from theia.types import Point, SituationalPicture

t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)


def get_empty_situational_picture() -> SituationalPicture:
    return SituationalPicture(
        time=t0,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


class TestCriticalInfrastructureController(unittest.TestCase):
    def test_update_produces_stationary_target(self):
        p = Point(lat=47.0, lon=8.0, alt=500.0)
        controller = CriticalInfrastructureController(
            target_id=42,
            name="Airport Zürich",
            point=p,
            sidc=SIDC.BLUE_GOVERNMENT_SITE,
            rcs=100.0,
        )

        controller.update(
            get_empty_situational_picture(), datetime.timedelta(seconds=1)
        )

        self.assertEqual(len(controller.targets), 1)
        target = controller.targets[0]
        self.assertEqual(target.id, 42)
        self.assertTrue(target.is_stationary)
        self.assertEqual(target.name, "Airport Zürich")
        self.assertEqual(target.point, p)
        self.assertEqual(target.cross_section_model.rcs, 100.0)
        self.assertEqual(
            (target.velocity.vx, target.velocity.vy, target.velocity.vz), (0, 0, 0)
        )

    def test_update_is_idempotent_and_stays_stationary(self):
        p = Point(lat=47.0, lon=8.0, alt=500.0)
        controller = CriticalInfrastructureController(
            target_id=1,
            name="Power Plant",
            point=p,
            sidc=SIDC.BLUE_GOVERNMENT_SITE,
            rcs=50.0,
        )

        for _ in range(3):
            controller.update(
                get_empty_situational_picture(), datetime.timedelta(seconds=1)
            )

        self.assertEqual(len(controller.targets), 1)
        target = controller.targets[0]
        self.assertEqual(target.id, 1)
        self.assertTrue(target.is_stationary)
        self.assertEqual(target.name, "Power Plant")
        self.assertEqual(target.sidc, SIDC.BLUE_GOVERNMENT_SITE.value)
        self.assertEqual(target.point, p)
        self.assertEqual(target.cross_section_model.rcs, 50.0)
        self.assertEqual(
            (target.velocity.vx, target.velocity.vy, target.velocity.vz), (0, 0, 0)
        )
        self.assertIsNone(target.receiver)
        self.assertIsNone(target.transmitter)


if __name__ == "__main__":
    unittest.main()
