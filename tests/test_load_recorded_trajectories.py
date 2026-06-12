import datetime
from pathlib import Path
import unittest


from theia.config import SIDC
from theia.coordinates import CoordinateTransformations
from theia.data_loading import load_trajectory_file
from theia.types import ConstantRcsModel, Point, Trajectory, Velocity


class TestLoadRecordedTrajectories(unittest.TestCase):
    def test_load_single_trajectory(self):
        trajectories, lookup = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/test_data/trajectories_single.csv"
        )
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(lookup[0], "HELLO")

        v_expected: list[Velocity] = [
            CoordinateTransformations.velocity_geodetic_to_cartesian(
                p=Point(lat=46.6155, lon=7.2784, alt=500.0),
                vlat=0.0,
                vlon=100.0,
                valt=20.0,
            ),
            CoordinateTransformations.velocity_geodetic_to_cartesian(
                p=Point(lat=46.6155, lon=7.3567, alt=620.0),
                vlat=0.0,
                vlon=100.0,
                valt=0.0,
            ),
        ]

        expected = Trajectory(
            target_id=0,
            target_sidc=SIDC.RED_FIXED_WING,
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
            vxs=[v_expected[0].vx, v_expected[1].vx],
            vys=[v_expected[0].vy, v_expected[1].vy],
            vzs=[v_expected[0].vz, v_expected[1].vz],
            cross_section_model=ConstantRcsModel(rcs=10.0),
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

        v_expected: list[Velocity] = [
            CoordinateTransformations.velocity_geodetic_to_cartesian(
                p=Point(lat=46.6155, lon=7.2784, alt=500.0),
                vlat=0.0,
                vlon=100.0,
                valt=20.0,
            ),
            CoordinateTransformations.velocity_geodetic_to_cartesian(
                p=Point(lat=46.6155, lon=7.3567, alt=620.0),
                vlat=0.0,
                vlon=100.0,
                valt=0.0,
            ),
        ]

        expected = Trajectory(
            target_id=0,
            target_sidc=SIDC.RED_FIXED_WING,
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
            vxs=[v_expected[0].vx, v_expected[1].vx],
            vys=[v_expected[0].vy, v_expected[1].vy],
            vzs=[v_expected[0].vz, v_expected[1].vz],
            cross_section_model=ConstantRcsModel(rcs=10.0),
        )
        self.assertEqual(trajectories[0], expected)

        v_expected: list[Velocity] = [
            CoordinateTransformations.velocity_geodetic_to_cartesian(
                p=Point(lat=47.7024, lon=8.6078, alt=1000.0),
                vlat=100,
                vlon=-50.0,
                valt=-13.0,
            ),
            CoordinateTransformations.velocity_geodetic_to_cartesian(
                p=Point(lat=46.6638, lon=7.3135, alt=987.0),
                vlat=100.0,
                vlon=0.0,
                valt=-13.0,
            ),
            CoordinateTransformations.velocity_geodetic_to_cartesian(
                p=Point(lat=46.6195, lon=7.2784, alt=928.5),
                vlat=100.0,
                vlon=0.0,
                valt=0.0,
            ),
        ]

        expected = Trajectory(
            target_id=1,
            target_sidc=SIDC.RED_FIXED_WING,
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
            vxs=[v_expected[0].vx, v_expected[1].vx, v_expected[2].vx],
            vys=[v_expected[0].vy, v_expected[1].vy, v_expected[2].vy],
            vzs=[v_expected[0].vz, v_expected[1].vz, v_expected[2].vz],
            cross_section_model=ConstantRcsModel(rcs=100.0),
        )
        self.assertEqual(trajectories[1], expected)


if __name__ == "__main__":
    unittest.main()
