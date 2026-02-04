import itertools
import unittest

import numpy as np

from theia.coordinates import CoordinateTransformations, calculate_elevation_angle
from theia.data_loading import elevationAt
from theia.distance import R_EARTH
from theia.types import Point


class ElevationAngleTest(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.Generator(np.random.PCG64(seed=467097))
        self._lats = np.linspace(-90, 90, 50)
        self._lons = np.linspace(-180, 180, 50)
        self._target_alts = np.linspace(0.1, 10_000.1, 50)

    def test_target_above_and_below_observer(self):
        for lat, lon, target_alt in itertools.product(
            self._lats,
            self._lons,
            self._target_alts,
        ):
            p_observer = Point(
                lat=lat,
                lon=lon,
                alt=self._rng.uniform(0.1, 10_000),
            )

            p_target = Point(
                lat=lat,
                lon=lon,
                alt=p_observer.alt + target_alt,
            )

            elevation = calculate_elevation_angle(p_observer, p_target)
            self.assertAlmostEqual(elevation, np.pi / 2, delta=0.005)

            p_target = Point(
                lat=lat,
                lon=lon,
                alt=p_observer.alt - target_alt,
            )
            elevation = calculate_elevation_angle(p_observer, p_target)
            self.assertAlmostEqual(elevation, -np.pi / 2, delta=0.005)

    def test_observer_at_north_pole(self):
        elevation_angles = np.linspace(-np.pi / 2, np.pi / 2, 50)
        p_observer = Point(lat=90, lon=0, alt=0)
        for longitude, elevation, alt in itertools.product(
            self._lons,
            elevation_angles,
            self._target_alts,
        ):
            R_observer = R_EARTH + p_observer.alt
            R_target = R_EARTH + alt

            lat_target = np.rad2deg(
                np.pi / 2
                - (np.arccos(R_observer / R_target * np.cos(elevation)) - elevation)
            )
            # lat_target = lat_target % 180
            p_target = Point(
                lat=lat_target,
                lon=longitude,
                alt=alt,
            )

            elevation_calculated = calculate_elevation_angle(p_observer, p_target)

            self.assertAlmostEqual(elevation, elevation_calculated, delta=0.005)


if __name__ == "__main__":
    unittest.main()
