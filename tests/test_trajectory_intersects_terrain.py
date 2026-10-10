import datetime
import unittest

from theia.terrain import AbstractTerrainModel, trajectory_intersects_terrain
from theia.types import ConstantRcsModel, TargetInfos, Trajectory


class RidgeTerrain(AbstractTerrainModel):
    """Flat terrain at 0 m with a ridge of given height between two latitudes."""

    lat_min: float
    lat_max: float
    height: float

    def elevationAt(self, lat: float, lon: float) -> float:
        return self.height if self.lat_min <= lat <= self.lat_max else 0.0


def _northbound_trajectory(lats: list[float], alts: list[float]) -> Trajectory:
    t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
    n = len(lats)
    return Trajectory(
        target_id=0,
        target_info=TargetInfos.FIXED_WING,
        times=[t0 + datetime.timedelta(seconds=10 * i) for i in range(n)],
        lats=lats,
        lons=[8.0] * n,
        alts=alts,
        vxs=[0.0] * n,
        vys=[0.0] * n,
        vzs=[0.0] * n,
        cross_section_model=ConstantRcsModel(rcs=1.0),
    )


class TrajectoryIntersectsTerrainTest(unittest.TestCase):
    def setUp(self):
        # Two waypoints ~11 km apart at 100 m altitude.
        self.trajectory = _northbound_trajectory([47.0, 47.1], [100.0, 100.0])

    def test_clear_trajectory(self):
        terrain = RidgeTerrain(lat_min=47.05, lat_max=47.051, height=50.0)
        self.assertFalse(trajectory_intersects_terrain(self.trajectory, terrain))

    def test_waypoint_below_terrain(self):
        terrain = RidgeTerrain(lat_min=47.09, lat_max=47.11, height=200.0)
        self.assertTrue(trajectory_intersects_terrain(self.trajectory, terrain))

    def test_waypoint_on_terrain(self):
        terrain = RidgeTerrain(lat_min=46.99, lat_max=47.01, height=100.0)
        self.assertTrue(trajectory_intersects_terrain(self.trajectory, terrain))

    def test_narrow_ridge_between_waypoints(self):
        # ~110 m wide ridge in the middle of the segment; both waypoints are
        # above the terrain, but the path between them is not.
        terrain = RidgeTerrain(lat_min=47.05, lat_max=47.051, height=200.0)
        self.assertTrue(trajectory_intersects_terrain(self.trajectory, terrain))

    def test_finer_step_finds_narrower_ridge(self):
        # ~11 m wide ridge: found with 5 m steps.
        terrain = RidgeTerrain(lat_min=47.05, lat_max=47.0501, height=200.0)
        self.assertTrue(
            trajectory_intersects_terrain(self.trajectory, terrain, step_m=5.0)
        )

    def test_descent_into_slope_before_target(self):
        # Descent from 300 m to a target 8 m above flat ground, with a 50 m
        # high ridge ~330 m before the target: the linear descent is at ~22 m
        # there, so it passes through the ridge.
        trajectory = _northbound_trajectory([47.0, 47.1], [300.0, 8.0])
        terrain = RidgeTerrain(lat_min=47.097, lat_max=47.098, height=50.0)
        self.assertTrue(trajectory_intersects_terrain(trajectory, terrain))


if __name__ == "__main__":
    unittest.main()
