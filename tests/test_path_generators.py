import datetime
import math
import unittest

import numpy as np

from theia.config import (
    DRONE_DEFAULT_CLIMB_ANGLE,
    DRONE_DEFAULT_CRUISE_ALT_MASL,
    DRONE_DEFAULT_DIVE_START_DISTANCE,
)
from theia.distance import haversine
from theia.simulation.path_generators import (
    BallisticGenerator,
    ClimbCruiseDiveGenerator,
    TerrainFollowingGenerator,
)
from theia.simulation.scenario_import import (
    BallisticMissileFactory,
    CruiseMissileFactory,
    TerrainFactory,
)
from theia.terrain import DummyTerrain
from theia.types import Point

T0 = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.UTC)
P_START = Point(lat=47.0, lon=8.0, alt=400.0)
P_STOP = Point(lat=47.0, lon=8.6, alt=500.0)
SPEED = 50.0


class ClimbCruiseDiveGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.path = ClimbCruiseDiveGenerator().generate(
            P_START, P_STOP, T0, SPEED, DummyTerrain(has_los=True)
        )
        self.alts = np.array(self.path.alts)
        # Ground distance still to fly [m].
        self.remaining = np.array(
            [
                haversine(lon, lat, P_STOP.lon, P_STOP.lat)
                for lat, lon in zip(self.path.lats, self.path.lons)
            ]
        )
        self.flown = np.array(
            [
                haversine(P_START.lon, P_START.lat, lon, lat)
                for lat, lon in zip(self.path.lats, self.path.lons)
            ]
        )

    def test_starts_at_launch_site_and_time(self):
        self.assertEqual(self.path.times[0], T0)
        self.assertAlmostEqual(self.path.lats[0], P_START.lat)
        self.assertAlmostEqual(self.path.lons[0], P_START.lon)
        self.assertAlmostEqual(self.path.alts[0], P_START.alt)

    def test_ends_at_destination(self):
        self.assertEqual(self.path.end_point, P_STOP)

    def test_arrives_at_whole_second(self):
        seconds = (self.path.times[-1] - T0).total_seconds()
        self.assertEqual(seconds, math.ceil(seconds))
        self.assertTrue(
            all(a < b for a, b in zip(self.path.times, self.path.times[1:]))
        )

    def test_climbs_at_climb_angle(self):
        climbing = self.alts < DRONE_DEFAULT_CRUISE_ALT_MASL - 1e-6
        climbing &= self.remaining > DRONE_DEFAULT_DIVE_START_DISTANCE
        self.assertGreater(climbing.sum(), 1)
        expected = P_START.alt + self.flown[climbing] * math.tan(
            math.radians(DRONE_DEFAULT_CLIMB_ANGLE)
        )
        np.testing.assert_allclose(self.alts[climbing], expected, atol=1.0)

    def test_cruises_at_5000_masl(self):
        self.assertAlmostEqual(self.alts.max(), DRONE_DEFAULT_CRUISE_ALT_MASL)
        self.assertEqual(DRONE_DEFAULT_CRUISE_ALT_MASL, 5000.0)

    def test_dives_from_1000_m_before_the_target(self):
        self.assertEqual(DRONE_DEFAULT_DIVE_START_DISTANCE, 1000.0)
        before_dive = self.remaining > DRONE_DEFAULT_DIVE_START_DISTANCE + 1.0
        diving = self.remaining < DRONE_DEFAULT_DIVE_START_DISTANCE - 1.0
        # The last sample before the dive is at cruise altitude ...
        self.assertAlmostEqual(
            self.alts[before_dive][-1], DRONE_DEFAULT_CRUISE_ALT_MASL, places=3
        )
        # ... and the dive is a straight line onto the destination.
        expected = P_STOP.alt + self.remaining[diving] * (
            (DRONE_DEFAULT_CRUISE_ALT_MASL - P_STOP.alt)
            / DRONE_DEFAULT_DIVE_START_DISTANCE
        )
        np.testing.assert_allclose(self.alts[diving], expected, atol=1.0)

    def test_constant_speed(self):
        seconds = np.array([(t - T0).total_seconds() for t in self.path.times])
        # Leave out the last interval, which is stretched to a whole second.
        dt = np.diff(seconds)[:-1]
        speeds = []
        for i in range(len(dt)):
            ds = haversine(
                self.path.lons[i],
                self.path.lats[i],
                self.path.lons[i + 1],
                self.path.lats[i + 1],
            )
            dz = self.path.alts[i + 1] - self.path.alts[i]
            speeds.append(math.hypot(ds, dz) / dt[i])
        np.testing.assert_allclose(speeds, SPEED, rtol=1e-3)

    def test_short_flight_skips_cruise(self):
        p_stop = Point(lat=47.0, lon=8.03, alt=400.0)
        path = ClimbCruiseDiveGenerator().generate(
            P_START, p_stop, T0, SPEED, DummyTerrain(has_los=True)
        )
        self.assertLess(max(path.alts), DRONE_DEFAULT_CRUISE_ALT_MASL)
        self.assertEqual(path.end_point, p_stop)

    def test_unreachable_destination(self):
        p_stop = Point(lat=47.0, lon=8.6, alt=6000.0)
        with self.assertRaises(ValueError):
            ClimbCruiseDiveGenerator().generate(
                P_START, p_stop, T0, SPEED, DummyTerrain(has_los=True)
            )


class BallisticGeneratorTest(unittest.TestCase):
    def test_matches_ballistic_missile_factory(self):
        p_start = Point(lat=47.0, lon=8.0, alt=0.0)
        p_stop = Point(lat=47.1, lon=8.1, alt=0.0)
        factory = BallisticMissileFactory(
            p_start=p_start,
            p_stop=p_stop,
            t_start=T0,
            terrain=TerrainFactory(terrain_name="SRTM"),
            target_id=1,
            effector_id=1,
            rcs=1.0,
            alpha=30.0,
        )
        expected = factory.get_trajectory(True)
        path = BallisticGenerator(alpha=30.0).generate(
            p_start, p_stop, T0, 0.0, DummyTerrain(has_los=True)
        )
        self.assertEqual(path.lats, expected.lats)
        self.assertEqual(path.lons, expected.lons)
        self.assertEqual(path.alts, expected.alts)
        self.assertEqual(path.times, expected.times)


class TerrainFollowingGeneratorTest(unittest.TestCase):
    def test_matches_cruise_missile_factory(self):
        terrain_factory = TerrainFactory(terrain_name="SRTM")
        p_start = Point(lat=47.0, lon=8.0, alt=0.0)
        p_stop = Point(lat=47.1, lon=8.1, alt=0.0)
        factory = CruiseMissileFactory(
            p_start=p_start,
            p_stop=p_stop,
            t_start=T0,
            terrain=terrain_factory,
            target_id=1,
            effector_id=1,
            rcs=0.1,
            speed=250.0,
            cruise_magl=80.0,
        )
        expected = factory.get_trajectory(True)
        path = TerrainFollowingGenerator(cruise_magl=80.0).generate(
            p_start, p_stop, T0, 250.0, terrain_factory.to_terrain()
        )
        self.assertEqual(path.times, expected.times)
        self.assertEqual(path.lats, expected.lats)
        self.assertEqual(path.lons, expected.lons)
        self.assertEqual(path.alts, expected.alts)
        self.assertEqual(path.end_point, factory.path.impact)


if __name__ == "__main__":
    unittest.main()
