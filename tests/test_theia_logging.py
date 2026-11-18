import datetime
import json
import os
import tempfile
import unittest

from theia.simulation.actions import (
    ActionExecutedEvent,
    ActionRejectedEvent,
    LaunchAction,
    MoveStockAction,
    RejectionReason,
)
from theia.simulation.theia_logging import FileLogger, LogLoader
from theia.types import (
    ConstantRcsModel,
    KillEvent,
    Party,
    Point,
    ShipmentArrivedEvent,
    ShipmentDepartedEvent,
    ShipmentLostEvent,
    SituationalPicture,
    Snapshot,
    StockLostEvent,
    Target,
    TargetInfos,
    TextEvent,
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


class LogLoaderEventsTest(unittest.TestCase):
    def test_events_keep_their_type(self):
        t = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
        shipment = dict(
            party=Party.RED,
            shipment_id=3,
            source_site_id=4,
            source_site_name="depot",
            target_site_id=5,
            target_site_name="launch site",
            count=6,
        )
        events = [
            KillEvent(id=1, time=t, target_id=7),
            ShipmentLostEvent(id=2, time=t, **shipment, destroyed_site_id=5),
            TextEvent(id=3, time=t, text="hello"),
            LaunchAction(
                id=4,
                time=t,
                party=Party.BLUE,
                site_id=1,
                destination=Point(lat=47.0, lon=8.0, alt=0.0),
            ),
            ActionExecutedEvent(
                id=5, time=t, party=Party.BLUE, action_id=4, result="launched"
            ),
            MoveStockAction(
                id=6,
                time=t,
                party=Party.RED,
                source_site_id=1,
                target_site_id=2,
                count=3,
            ),
            ActionRejectedEvent(
                id=7,
                time=t,
                party=Party.RED,
                action_id=6,
                reason=RejectionReason.NO_ROUTE,
                message="no route from camp site 1 to 2",
            ),
            ShipmentDepartedEvent(id=8, time=t, **shipment),
            ShipmentArrivedEvent(id=9, time=t, **shipment, n_ammo=5),
            StockLostEvent(
                id=10, time=t, party=Party.RED, site_id=4, site_name="depot", count=2
            ),
        ]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, "run.log.json")
        logger = FileLogger(path)
        logger.on_events(events)
        logger.on_end()
        loader = LogLoader(path)
        self.assertEqual(loader._events, events)
        self.assertEqual(loader.shipment_lost_events, [events[1]])
        self.assertEqual(loader.kill_events, [events[0]])
        self.assertEqual(loader.text_events, [events[2]])
        self.assertEqual(loader.actions, [events[3], events[5]])
        self.assertEqual(loader.action_executed_events, [events[4]])
        self.assertEqual(loader.action_rejected_events, [events[6]])
        self.assertEqual(loader.shipment_events, [events[1], events[7], events[8]])
        self.assertEqual(loader.stock_lost_events, [events[9]])


if __name__ == "__main__":
    unittest.main()
