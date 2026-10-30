import datetime
import json
import os
import tempfile
import unittest

from theia.simulation.theia_logging import FileLogger, LogLoader
from theia.types import (
    ConstantRcsModel,
    Party,
    Point,
    SituationalPicture,
    Snapshot,
    Target,
    TargetInfos,
    Velocity,
)


def _target(target_id: int, party: Party, t: float) -> Target:
    return Target(
        id=target_id,
        is_stationary=False,
        name="Zürich",  # non-ASCII: json.dumps escapes it
        info=TargetInfos.FIXED_WING.with_party(party),
        point=Point(lat=47.0 + 0.01 * t, lon=8.0, alt=1_000.0),
        cross_section_model=ConstantRcsModel(rcs=1.0),
        velocity=Velocity(vx=float("inf"), vy=float("nan"), vz=0.0),
    )


def _snapshot(t: float) -> Snapshot:
    return Snapshot(
        time=datetime.datetime.fromtimestamp(t, datetime.UTC),
        blue_monostatic_radars=[],
        blue_pcl_sensors=[],
        blue_targets=[_target(1, Party.BLUE, t)],
        red_monostatic_radars=[],
        red_pcl_sensors=[],
        red_targets=[_target(2, Party.RED, t), _target(3, Party.RED, t)],
    )


def _situational_picture(t: float) -> SituationalPicture:
    return SituationalPicture(
        time=datetime.datetime.fromtimestamp(t, datetime.UTC),
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[_target(1, Party.BLUE, t)],
        enemy_targets=[],
    )


class FileLoggerTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = os.path.join(directory.name, "run.log.json")
        self.snapshots = [_snapshot(t) for t in (0.0, 1.0, 2.0)]

    def _write(self, log_situational_picture: bool) -> str:
        logger = FileLogger(self.path, log_situational_picture=log_situational_picture)
        for i, snapshot in enumerate(self.snapshots):
            logger.on_situational_picture(_situational_picture(i), is_blue=True)
            logger.on_snapshot(snapshot)
        logger.on_end()
        with open(self.path) as file:
            return file.read()

    def test_identical_to_dumping_everything_at_once(self):
        # The file format must not depend on how the logger buffers snapshots.
        # This includes inf/NaN, which model_dump_json would turn into null.
        expected = json.dumps(
            {
                "snapshots": [s.model_dump(mode="json") for s in self.snapshots],
                "detections": [],
                "events": [],
            }
        )
        self.assertEqual(self._write(log_situational_picture=False), expected)

    def test_identical_with_situational_pictures(self):
        expected = json.dumps(
            {
                "snapshots": [s.model_dump(mode="json") for s in self.snapshots],
                "detections": [],
                "events": [],
                "situational_pictures": [
                    {
                        "team": "blue",
                        "situational_picture": _situational_picture(i).model_dump(
                            mode="json"
                        ),
                    }
                    for i in range(len(self.snapshots))
                ],
            }
        )
        self.assertEqual(self._write(log_situational_picture=True), expected)

    def test_situational_pictures_not_kept_unless_logged(self):
        logger = FileLogger(self.path, log_situational_picture=False)
        logger.on_situational_picture(_situational_picture(0.0), is_blue=True)
        self.assertEqual(logger.situational_pictures, [])

    def test_without_snapshots(self):
        logger = FileLogger(self.path)
        logger.on_end()
        with open(self.path) as file:
            data = json.load(file)
        self.assertEqual(data, {"snapshots": [], "detections": [], "events": []})


class LogLoaderTargetIdsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, "run.log.json")
        logger = FileLogger(path)
        for t in (0.0, 1.0, 2.0):
            logger.on_snapshot(_snapshot(t))
        logger.on_end()
        self.loader = LogLoader(path)

    def test_ids(self):
        self.assertEqual(self.loader.blue_target_ids, {1})
        self.assertEqual(self.loader.red_target_ids, {2, 3})

    def test_ids_are_ground_truth_keys(self):
        self.assertEqual(
            self.loader.blue_target_ids,
            set(self.loader.blue_target_ground_truth.keys()),
        )
        self.assertEqual(
            self.loader.red_target_ids,
            set(self.loader.red_target_ground_truth.keys()),
        )


if __name__ == "__main__":
    unittest.main()
