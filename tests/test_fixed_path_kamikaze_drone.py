import datetime
import unittest
from unittest.mock import MagicMock, patch

from theia.simulation.controllers.fixed_path_kamikaze_drone import FixedPathOneWayDrone
from theia.types import DirectShot, KillEvent, Party, TargetInfos, Trajectory

T0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
DRONE_TARGET_ID = 5


def make_drone() -> FixedPathOneWayDrone:
    trajectory = Trajectory.model_construct(
        target_id=DRONE_TARGET_ID,
        target_info=TargetInfos.FIXED_WING.with_party(Party.RED),
    )
    return FixedPathOneWayDrone.model_construct(
        effector=MagicMock(id=11), trajectory=trajectory
    )


def make_shot(shooter_id: int) -> MagicMock:
    shot = MagicMock(spec=DirectShot)
    shot.shooter = MagicMock(id=shooter_id)
    shot.time = T0
    return shot


class FixedPathOneWayDroneTest(unittest.TestCase):
    @patch.object(FixedPathOneWayDrone, "_broadcast_event")
    def test_own_shot_kills_the_drone_with_its_party(self, broadcast):
        make_drone().on_event(make_shot(shooter_id=11))

        broadcast.assert_called_once()
        event = broadcast.call_args[0][0]
        self.assertIsInstance(event, KillEvent)
        self.assertEqual(event.time, T0)
        self.assertEqual(event.target_id, DRONE_TARGET_ID)
        # The drone itself is the victim, so the event belongs to its side.
        self.assertEqual(event.party, Party.RED)

    @patch.object(FixedPathOneWayDrone, "_broadcast_event")
    def test_other_shot_is_ignored(self, broadcast):
        make_drone().on_event(make_shot(shooter_id=12))

        broadcast.assert_not_called()


if __name__ == "__main__":
    unittest.main()
