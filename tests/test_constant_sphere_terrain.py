import unittest

from theia.terrain import ConstantSphereTerrainModel
from theia.types import Point


class ConstantSphereLineOfSightTest(unittest.TestCase):
    def setUp(self):
        self.terrain = ConstantSphereTerrainModel(alt=400.0)

    def test_nearby_points_above_the_terrain_see_each_other(self):
        p1 = Point(lat=47.0, lon=8.0, alt=500.0)
        p2 = Point(lat=47.01, lon=8.0, alt=500.0)

        self.assertTrue(self.terrain.has_line_of_sight(p1, p2))

    def test_ground_radar_sees_a_high_aircraft(self):
        # Regression test: the closest point of the infinite line lies behind
        # the radar, below the terrain; the segment itself is clear.
        radar = Point(lat=47.0, lon=8.0, alt=500.0)
        aircraft = Point(lat=47.9, lon=8.0, alt=10_000.0)

        self.assertTrue(self.terrain.has_line_of_sight(radar, aircraft))
        self.assertTrue(self.terrain.has_line_of_sight(aircraft, radar))

    def test_earth_curvature_blocks_the_line_of_sight(self):
        # 500 m above the terrain, the horizon is ~80 km away: 400 km is beyond.
        p1 = Point(lat=47.0, lon=8.0, alt=500.0)
        p2 = Point(lat=50.6, lon=8.0, alt=500.0)

        self.assertFalse(self.terrain.has_line_of_sight(p1, p2))

    def test_point_below_the_terrain_has_no_line_of_sight(self):
        p1 = Point(lat=47.0, lon=8.0, alt=300.0)
        p2 = Point(lat=47.01, lon=8.0, alt=500.0)

        self.assertFalse(self.terrain.has_line_of_sight(p1, p2))


if __name__ == "__main__":
    unittest.main()
