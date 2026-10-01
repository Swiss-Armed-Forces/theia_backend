import datetime
import unittest

import pydantic

from theia.types import (
    ConstantRcsModel,
    Party,
    Point,
    Target,
    TargetCategory,
    TargetInfo,
    TargetInfos,
    Track,
    Trajectory,
    Velocity,
)


def get_target(info: TargetInfo, is_damaged: bool = False) -> Target:
    return Target(
        id=1,
        is_stationary=True,
        info=info,
        is_damaged=is_damaged,
        point=Point(lat=47.0, lon=8.0, alt=500.0),
        cross_section_model=ConstantRcsModel(rcs=1.0),
        velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
    )


class TargetInfoTest(unittest.TestCase):
    def test_sidc_per_party(self):
        info = TargetInfos.GBAD
        self.assertEqual(info.with_party(Party.UNKNOWN).sidc(), "10011000001301000000")
        self.assertEqual(info.with_party(Party.BLUE).sidc(), "10031000001301000000")
        self.assertEqual(info.with_party(Party.NEUTRAL).sidc(), "10041000001301000000")
        self.assertEqual(info.with_party(Party.RED).sidc(), "10061000001301000000")

    def test_damaged_sidc(self):
        info = TargetInfos.GBAD.with_party(Party.BLUE)
        self.assertEqual(info.sidc(damaged=True), "10031030001301000000")

    def test_from_is_blue(self):
        self.assertEqual(Party.from_is_blue(True), Party.BLUE)
        self.assertEqual(Party.from_is_blue(False), Party.RED)

    def test_with_party_keeps_catalog_entry_unchanged(self):
        blue = TargetInfos.RADAR.with_party(Party.BLUE)
        self.assertEqual(blue.party, Party.BLUE)
        self.assertEqual(TargetInfos.RADAR.party, Party.UNKNOWN)
        self.assertEqual(blue.category, TargetCategory.SENSOR)

    def test_invalid_template(self):
        with self.assertRaises(pydantic.ValidationError):
            TargetInfo(sidc_template="10031000001301000000")
        with self.assertRaises(pydantic.ValidationError):
            TargetInfo(sidc_template="100x")

    def test_from_sidc(self):
        info = TargetInfo.from_sidc("10261500002203000000")
        self.assertEqual(info.sidc_template, "102x1500002203000000")
        self.assertEqual(info.party, Party.RED)
        self.assertEqual(info.category, TargetCategory.UNKNOWN)
        self.assertEqual(info.sidc(), "10261500002203000000")

    def test_radar_and_pcl_receiver_are_distinguishable(self):
        radar = TargetInfos.RADAR.with_party(Party.BLUE)
        receiver = TargetInfos.PCL_RECEIVER.with_party(Party.BLUE)
        self.assertEqual(radar.sidc(), receiver.sidc())
        self.assertNotEqual(radar, receiver)
        self.assertEqual(radar.category, TargetCategory.SENSOR)
        self.assertEqual(receiver.category, TargetCategory.UNKNOWN)
        self.assertIn("pcl_receiver", receiver.tags)

    def test_for_category(self):
        for category in TargetCategory:
            info = TargetInfos.for_category(category)
            self.assertEqual(info.category, category)

    def test_json_round_trip(self):
        info = TargetInfos.CRITICAL_INFRASTRUCTURE.with_party(Party.RED)
        self.assertEqual(
            TargetInfo.model_validate_json(info.model_dump_json()),
            info,
        )


class TargetTest(unittest.TestCase):
    def test_category_and_sidc(self):
        target = get_target(TargetInfos.GBAD.with_party(Party.BLUE))
        self.assertEqual(target.category, TargetCategory.GBAD)
        self.assertEqual(target.sidc, "10031000001301000000")

    def test_damaged(self):
        target = get_target(TargetInfos.GBAD.with_party(Party.BLUE), is_damaged=True)
        self.assertEqual(target.sidc, "10031030001301000000")

    def test_json_round_trip(self):
        target = get_target(TargetInfos.RADAR.with_party(Party.RED), is_damaged=True)
        data = target.model_dump(mode="json")
        self.assertEqual(data["sidc"], "10261530002203000000")
        self.assertEqual(data["info"]["category"], "SENSOR")
        restored = Target.model_validate(data)
        self.assertEqual(restored.info, target.info)
        self.assertTrue(restored.is_damaged)

    def test_legacy_sidc(self):
        data = get_target(TargetInfos.UNKNOWN).model_dump(mode="json")
        del data["info"]
        del data["is_damaged"]
        data["sidc"] = "10061030001301000000"
        target = Target.model_validate(data)
        self.assertEqual(target.info.party, Party.RED)
        self.assertEqual(target.category, TargetCategory.UNKNOWN)
        self.assertTrue(target.is_damaged)
        self.assertEqual(target.sidc, "10061030001301000000")


class TrajectoryTest(unittest.TestCase):
    def get_data(self) -> dict:
        t0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
        return {
            "target_id": 3,
            "times": [t0.isoformat(), (t0 + datetime.timedelta(seconds=1)).isoformat()],
            "lats": [47.0, 47.0],
            "lons": [8.0, 8.0],
            "alts": [500.0, 500.0],
            "vxs": [0.0, 0.0],
            "vys": [0.0, 0.0],
            "vzs": [0.0, 0.0],
            "cross_section_model": {"rcs": 1.0},
        }

    def test_target_info_propagates_to_target(self):
        data = self.get_data()
        data["target_info"] = TargetInfos.for_category(
            TargetCategory.DRONE_CLASS_II
        ).model_dump(mode="json")
        trajectory = Trajectory.model_validate(data)
        target = trajectory(trajectory.times[0])
        self.assertEqual(target.category, TargetCategory.DRONE_CLASS_II)

    def test_legacy_target_sidc(self):
        data = self.get_data()
        data["target_sidc"] = "10260100001101000000"
        trajectory = Trajectory.model_validate(data)
        self.assertEqual(trajectory.target_info.party, Party.RED)
        self.assertEqual(trajectory(trajectory.times[0]).sidc, "10260100001101000000")


class TrackTest(unittest.TestCase):
    def test_sidc(self):
        t0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
        track = Track(
            id="1",
            target_info=TargetInfos.FIXED_WING.with_party(Party.RED),
            states=[
                (t0, [0.0] * 6),
                (t0 + datetime.timedelta(seconds=1), [1.0] * 6),
            ],
        )
        self.assertEqual(track.sidc, "10260100001101000000")


if __name__ == "__main__":
    unittest.main()
