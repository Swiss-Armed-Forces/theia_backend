import datetime
import types
import unittest

from fastapi.testclient import TestClient

from theia.simulation.server import create_app
from theia.simulation.theia_logging import SituationalPictureBuffer
from theia.types import (
    ConstantRcsModel,
    Party,
    Point,
    SituationalPicture,
    Target,
    TargetInfos,
    Track,
    Velocity,
)

T0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)


def get_client() -> TestClient:
    buffer = SituationalPictureBuffer()
    target = Target(
        id=7,
        is_stationary=True,
        name="GBAD",
        info=TargetInfos.GBAD.with_party(Party.BLUE),
        point=Point(lat=47.0, lon=8.0, alt=500.0),
        cross_section_model=ConstantRcsModel(rcs=1.0),
        velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
    )
    buffer.on_snapshot(
        types.SimpleNamespace(time=T0, blue_targets=[target], red_targets=[])
    )
    track = Track(
        id="42",
        target_info=TargetInfos.FIXED_WING.with_party(Party.RED),
        states=[
            (T0, [4.3e6, 0.0, 6.0e5, 0.0, 4.6e6, 0.0]),
            (T0 + datetime.timedelta(seconds=1), [4.3e6, 0.0, 6.0e5, 0.0, 4.6e6, 0.0]),
        ],
    )
    buffer.on_situational_picture(
        SituationalPicture(
            time=T0,
            friendly_pet_receivers=[],
            friendly_radars=[],
            friendly_targets=[],
            enemy_targets=[track],
        ),
        is_blue=True,
    )
    return TestClient(create_app(buffer, director=None))


class ServerApiTest(unittest.TestCase):
    def test_ground_truth_contains_info(self):
        response = get_client().post("/ground_truth/BLUE", json=[T0.isoformat()])
        self.assertEqual(response.status_code, 200)
        [gt] = response.json()
        self.assertEqual(gt["sidc"], "10031000001301000000")
        self.assertEqual(gt["info"]["category"], "GBAD")
        self.assertEqual(gt["info"]["party"], "BLUE")

    def test_situational_picture_contains_info(self):
        response = get_client().post("/situational_picture/BLUE", json=[T0.isoformat()])
        self.assertEqual(response.status_code, 200)
        [track] = response.json()["enemy_tracks"]
        self.assertEqual(track["sidc"], "10260100001101000000")
        self.assertEqual(track["info"]["party"], "RED")

    def test_openapi_schema_exposes_target_info(self):
        schemas = create_app(None, None).openapi()["components"]["schemas"]
        self.assertIn("TargetInfo", schemas)
        self.assertIn("TargetCategory", schemas)


if __name__ == "__main__":
    unittest.main()
