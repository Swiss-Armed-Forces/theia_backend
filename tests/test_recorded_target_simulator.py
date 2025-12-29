import datetime
from pathlib import Path
import unittest

from theia.data_loading import load_trajectory_file
from theia.target_simulation.recorded_targets_simulator import RecordedTargetsSimulator
from theia.types import Point, Target


class RecordedTargetSimulatorTest(unittest.TestCase):
    def setUp(self):
        trajectories, _ = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/test_data/trajectories_multiple.csv"
        )
        self.simulator = RecordedTargetsSimulator(trajectories)

        trajectories, _ = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/test_data/trajectories_multiple_simultaneous.csv"
        )
        self.simulator2 = RecordedTargetsSimulator(trajectories)

    def test_out_of_range(self):
        targets = self.simulator.get_targets(
            datetime.datetime(
                year=2022,
                month=1,
                day=1,
            )
        )
        self.assertEqual(len(targets), 0)
        targets = self.simulator.get_targets(
            datetime.datetime(
                year=2026,
                month=1,
                day=1,
            )
        )
        self.assertEqual(len(targets), 0)

    def test_endpoints(self):
        # Start point.
        targets = self.simulator.get_targets(
            datetime.datetime(
                year=2022,
                month=6,
                day=27,
                hour=4,
                minute=2,
                second=23,
            )
        )
        expected = Target(
            id=0,
            point=Point(
                lat=46.6155,
                lon=7.2784,
                alt=500.0,
            ),
            cross_section=10.0,
            vlat=0.0,
            vlon=100.0,
            vz=20.0,
        )
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], expected)

        # End point.
        targets = self.simulator.get_targets(
            datetime.datetime(
                year=2022,
                month=6,
                day=27,
                hour=4,
                minute=3,
                second=23,
            )
        )
        expected = Target(
            id=0,
            point=Point(
                lat=46.6155,
                lon=7.3567,
                alt=620.0,
            ),
            cross_section=10.0,
            vlat=0.0,
            vlon=100.0,
            vz=0.0,
        )
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], expected)

    def test_between(self):
        # Halfway between the waypoints.
        targets = self.simulator.get_targets(
            datetime.datetime(
                year=2022,
                month=6,
                day=27,
                hour=4,
                minute=2,
                second=53,
            )
        )
        expected = Target(
            id=0,
            point=Point(
                lat=46.6155,
                lon=(7.2784 + 7.3567) * 0.5,
                alt=560.0,
            ),
            cross_section=10.0,
            vlat=0.0,
            vlon=100.0,
            vz=10.0,
        )
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], expected)

    def test_multiple_simultaneous_targets(self):
        targets = self.simulator2.get_targets(
            datetime.datetime(
                year=2022,
                month=6,
                day=27,
                hour=4,
                minute=2,
                second=53,
            )
        )
        expected = [
            Target(
                id=0,
                point=Point(
                    lat=46.6155,
                    lon=(7.2784 + 7.3567) * 0.5,
                    alt=560.0,
                ),
                cross_section=10.0,
                vlat=0.0,
                vlon=100.0,
                vz=10.0,
            ),
            Target(
                id=1,
                point=Point(
                    lat=47.7024,
                    lon=8.6078,
                    alt=1000.0,
                ),
                cross_section=100.0,
                vlat=100.0,
                vlon=-50.0,
                vz=-13.0,
            ),
        ]
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets, expected)

    def test_min_max_time(self):
        self.assertEqual(
            self.simulator.get_minimum_time(),
            datetime.datetime(
                year=2022,
                month=6,
                day=27,
                hour=4,
                minute=2,
                second=23,
            ),
        )
        self.assertEqual(
            self.simulator.get_maximum_time(),
            datetime.datetime(
                year=2022,
                month=7,
                day=23,
                hour=6,
                minute=19,
                second=37,
            ),
        )

        self.assertEqual(
            self.simulator2.get_maximum_time(),
            datetime.datetime(
                year=2022,
                month=6,
                day=27,
                hour=4,
                minute=15,
                second=0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
