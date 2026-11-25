import datetime
import unittest

from theia.types import KillEvent, Party, TrackInitEvent, name_and_id

T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


class NameAndIdTest(unittest.TestCase):
    def test_with_name(self):
        self.assertEqual(name_and_id("Uetliberg", 7), "Uetliberg (#7)")

    def test_without_name(self):
        self.assertEqual(name_and_id("", 7), "#7")


class KillEventTest(unittest.TestCase):
    def test_str_uses_target_name(self):
        event = KillEvent(
            id=1, time=T0, target_id=17, party=Party.RED, target_name="Shahed"
        )
        self.assertEqual(str(event), "Shahed (#17) 💀")

    def test_str_without_target_name(self):
        event = KillEvent(id=1, time=T0, target_id=17, party=Party.RED)
        self.assertEqual(str(event), "#17 💀")

    def test_legacy_log_without_name_loads(self):
        event = KillEvent.model_validate(
            {"id": 1, "time": T0.isoformat(), "target_id": 17}
        )
        self.assertEqual(event.target_name, "")
        self.assertEqual(event.party, Party.UNKNOWN)


class TrackInitEventTest(unittest.TestCase):
    def test_str(self):
        event = TrackInitEvent(id=1, time=T0, track_id=3)
        self.assertEqual(str(event), "Track #3 initialized")

    def test_carries_no_target_information(self):
        self.assertEqual(set(TrackInitEvent.model_fields), {"id", "time", "track_id"})

    def test_legacy_log_with_target_id_loads(self):
        event = TrackInitEvent.model_validate(
            {"id": 1, "time": T0.isoformat(), "target_id": 17, "track_id": 3}
        )
        self.assertEqual(str(event), "Track #3 initialized")


if __name__ == "__main__":
    unittest.main()
