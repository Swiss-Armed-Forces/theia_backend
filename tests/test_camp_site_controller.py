import datetime
import unittest

from theia.simulation.controllers.camp_site_controller import CampSiteController
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.scenario_import import CampSite, OneWayDroneTemplate
from theia.types import KillEvent, Party, Point, SituationalPicture

T0 = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.UTC)
DT = datetime.timedelta(seconds=1)
TARGET_ID = 10


def build_camp_site() -> CampSite:
    return CampSite(
        id=1,
        name="launch site",
        point=Point(lat=47.0, lon=8.0, alt=1.0),
        target_id=TARGET_ID,
        rcs=10.0,
        n_ammo=2,
        ammo=OneWayDroneTemplate(rcs=0.1, speed=50.0),
        can_launch=True,
        seconds_between_launches=30.0,
    )


class CampSiteControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = CampSiteController(camp_site=build_camp_site(), is_blue=True)

    def update(self):
        self.controller.update(
            SituationalPicture(
                time=T0,
                friendly_pet_receivers=[],
                friendly_radars=[],
                friendly_targets=[],
                enemy_targets=[],
            ),
            DT,
        )

    def test_alive_site_is_a_target(self):
        self.update()
        [target] = self.controller.targets
        self.assertEqual(target.id, TARGET_ID)
        self.assertEqual(target.name, "launch site")
        self.assertTrue(target.is_stationary)
        self.assertEqual(target.info.party, Party.BLUE)
        self.assertIn("launch_site", target.info.tags)

    def test_kill_event_is_left_to_the_living_controller(self):
        self.controller.on_event(KillEvent(id=1, time=T0, target_id=TARGET_ID))
        self.update()
        [target] = self.controller.targets
        self.assertEqual(target.id, TARGET_ID)

    def test_seconds_until_ready(self):
        self.assertEqual(self.controller.seconds_until_ready(T0), 0.0)
        self.controller.time_of_last_launch = T0
        self.assertEqual(self.controller.seconds_until_ready(T0), 30.0)
        self.assertEqual(
            self.controller.seconds_until_ready(T0 + datetime.timedelta(seconds=20)),
            10.0,
        )
        self.assertEqual(
            self.controller.seconds_until_ready(T0 + datetime.timedelta(seconds=40)),
            0.0,
        )


class CampSiteToControllerTest(unittest.TestCase):
    def test_wraps_a_copy_in_a_living_controller(self):
        camp_site = build_camp_site()
        living = camp_site.to_controller(is_blue=False)
        self.assertIsInstance(living, LivingController)
        self.assertEqual(living.target_id, TARGET_ID)
        self.assertFalse(living.child.is_blue)
        self.assertEqual(living.child.camp_site, camp_site)
        living.child.camp_site.n_ammo -= 1
        self.assertEqual(camp_site.n_ammo, 2)


if __name__ == "__main__":
    unittest.main()
