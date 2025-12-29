import datetime
from pathlib import Path
import unittest

import numpy as np

from theia.data_loading import load_trajectory_file
from theia.types import Trajectory


class TestLoadRecordedTrajectories(unittest.TestCase):
    def test_load_single_trajectory(self):
        trajectories, lookup = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/test_data/trajectories_single.csv"
        )
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(lookup[0], "HELLO")

        expected = Trajectory(
            target_id=0,
            times=[
                datetime.datetime(
                    year=2022, month=6, day=27, hour=4, minute=2, second=23
                ),
                datetime.datetime(
                    year=2022, month=6, day=27, hour=4, minute=3, second=23
                ),
            ],
            lats=[46.6155, 46.6155],
            lons=[7.2784, 7.3567],
            alts=[500.0, 620.0],
            vlats=[0.0, 0.0],
            vlons=[100.0, 100.0],
            vzs=[20.0, 0.0],
            cross_sections=[10.0, 10.0],
        )
        self.assertEqual(trajectories[0], expected)

    def test_load_multiple_trajectories(self):
        trajectories, lookup = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/test_data/trajectories_multiple.csv"
        )
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(len(lookup), 2)
        self.assertEqual(lookup[0], "HELLO")
        self.assertEqual(lookup[1], "HOHOHO")

        expected = Trajectory(
            target_id=0,
            times=[
                datetime.datetime(
                    year=2022,
                    month=6,
                    day=27,
                    hour=4,
                    minute=2,
                    second=23,
                ),
                datetime.datetime(
                    year=2022,
                    month=6,
                    day=27,
                    hour=4,
                    minute=3,
                    second=23,
                ),
            ],
            lats=[46.6155, 46.6155],
            lons=[7.2784, 7.3567],
            alts=[500.0, 620.0],
            vlats=[0.0, 0.0],
            vlons=[100.0, 100.0],
            vzs=[20.0, 0.0],
            cross_sections=[10.0, 10.0],
        )
        self.assertEqual(trajectories[0], expected)

        expected = Trajectory(
            target_id=1,
            times=[
                datetime.datetime(
                    year=2022,
                    month=7,
                    day=23,
                    hour=6,
                    minute=14,
                    second=7,
                ),
                datetime.datetime(
                    year=2022,
                    month=7,
                    day=23,
                    hour=6,
                    minute=15,
                    second=7,
                ),
                datetime.datetime(
                    year=2022,
                    month=7,
                    day=23,
                    hour=6,
                    minute=19,
                    second=37,
                ),
            ],
            lats=[47.7024, 46.6638, 46.6195],
            lons=[8.6078, 7.3135, 7.2784],
            alts=[1000.0, 987, 928.5],
            vlats=[100.0, 100.0, 100.0],
            vlons=[-50.0, 0.0, 0.0],
            vzs=[-13.0, -13.0, 0.0],
            cross_sections=[100.0, 100.0, 100.0],
        )
        self.assertEqual(trajectories[1], expected)


if __name__ == "__main__":
    unittest.main()
