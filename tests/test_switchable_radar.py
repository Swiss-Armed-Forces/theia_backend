import datetime
import unittest

from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.switchable_radar import (
    SwitchableMonostaticRadarController,
)
from theia.simulation.simulator import KillEvent
from theia.test_data import get_uetliberg_radar
from theia.types import (
    ConstantRcsModel,
    GeoJSONAnnotation,
    GeoJSONFeature,
    GeoJSONKind,
    GeoJSONPolygon,
    SituationalPicture,
)

TARGET_ID = 12
T0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
DT = datetime.timedelta(seconds=1)


def get_picture() -> SituationalPicture:
    return SituationalPicture(
        time=T0,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


class RadarCoverageTest(unittest.TestCase):
    def setUp(self):
        radar = get_uetliberg_radar()
        self._coverage = GeoJSONAnnotation(
            id="sensor-range",
            feature=GeoJSONFeature(
                geometry=GeoJSONPolygon(coordinates=[[[0.0, 0.0]]])
            ),
            kind=GeoJSONKind.SENSOR_RANGE,
            target_ids=[TARGET_ID],
        )
        self._switchable = SwitchableMonostaticRadarController(
            child=MonostaticRadarController(
                target_id=TARGET_ID,
                radar=radar,
                is_blue=True,
                rcs_model=ConstantRcsModel(rcs=1.0),
                coverage=[self._coverage],
            ),
            sensor_id=radar.id,
        )
        self._controller = LivingController(
            child=self._switchable, target_id=TARGET_ID
        )

    def test_coverage_published_while_on_and_alive(self):
        self._controller.update(get_picture(), DT)

        self.assertEqual(self._controller.geojson, [self._coverage])

    def test_coverage_hidden_while_off(self):
        self._switchable.is_on = False
        self._controller.update(get_picture(), DT)
        self.assertEqual(self._controller.geojson, [])

        self._switchable.is_on = True
        self._controller.update(get_picture(), DT)
        self.assertEqual(self._controller.geojson, [self._coverage])

    def test_coverage_hidden_when_dead(self):
        self._controller.on_event(KillEvent(id=1, time=T0, target_id=TARGET_ID))
        self._controller.update(get_picture(), DT)

        self.assertEqual(self._controller.geojson, [])


if __name__ == "__main__":
    unittest.main()
