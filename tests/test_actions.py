import unittest

import pydantic

from theia.config import UNKNOWN_ID, UNKNOWN_TIME
from theia.simulation.actions import (
    Action,
    AnyAction,
    LaunchAction,
    MoveStockAction,
    ToggleMonostaticRadarAction,
)
from theia.simulation.path_generators import (
    BallisticGenerator,
    ClimbCruiseDiveGenerator,
)
from theia.types import Event, Party, Point

ACTION = pydantic.TypeAdapter(AnyAction)


class ActionTest(unittest.TestCase):
    def test_round_trip(self):
        actions = [
            LaunchAction(
                site_id=1,
                destination=Point(lat=47.0, lon=8.0, alt=400.0),
            ),
            LaunchAction(
                site_id=1,
                destination=Point(lat=47.0, lon=8.0, alt=400.0),
                path_generator=BallisticGenerator(alpha=30.0),
            ),
            LaunchAction(
                site_id=1,
                destination=Point(lat=47.0, lon=8.0, alt=400.0),
                path_generator=ClimbCruiseDiveGenerator(cruise_alt_masl=3000.0),
            ),
            MoveStockAction(source_site_id=1, target_site_id=2, count=3),
            ToggleMonostaticRadarAction(sensor_id=4, new_is_on=False),
        ]
        for action in actions:
            with self.subTest(action=action):
                parsed = ACTION.validate_json(ACTION.dump_json(action))
                self.assertIs(type(parsed), type(action))
                self.assertEqual(parsed, action)

    def test_discriminator_picks_the_right_class(self):
        # A launch and a move share field names and value types; only the
        # discriminator tells them apart.
        launch = ACTION.validate_python(
            {
                "type": "launch",
                "site_id": 2,
                "destination": {"lat": 47.0, "lon": 8.0, "alt": 0.0},
            }
        )
        move = ACTION.validate_python(
            {
                "type": "move_stock",
                "source_site_id": 2,
                "target_site_id": 2,
                "count": 1,
            }
        )
        self.assertIsInstance(launch, LaunchAction)
        self.assertIsNone(launch.path_generator)
        self.assertIsInstance(move, MoveStockAction)

    def test_path_generator_discriminator(self):
        launch = ACTION.validate_python(
            {
                "type": "launch",
                "site_id": 2,
                "destination": {"lat": 47.0, "lon": 8.0, "alt": 0.0},
                "path_generator": {"type": "ballistic", "alpha": 60.0},
            }
        )
        self.assertEqual(launch.path_generator, BallisticGenerator(alpha=60.0))

    def test_actions_are_events(self):
        action = ToggleMonostaticRadarAction(sensor_id=4, new_is_on=False)
        self.assertIsInstance(action, Action)
        self.assertIsInstance(action, Event)
        # Filled in by the ActionController on execution.
        self.assertEqual(action.id, UNKNOWN_ID)
        self.assertEqual(action.time, UNKNOWN_TIME)
        self.assertEqual(action.party, Party.UNKNOWN)

    def test_unknown_type_is_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            ACTION.validate_python({"type": "idle"})


if __name__ == "__main__":
    unittest.main()
