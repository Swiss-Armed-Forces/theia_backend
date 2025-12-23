from pathlib import Path
import unittest

import geopandas as gpd
from pyproj import Transformer
import shapely

from theia.coverage import calculate_coverage
from theia.data_loading import elevationAt
from theia.radar_equation import radar_eq_max_dist
from theia.types import Point, Polarization, Radar


class CoverageTest(unittest.TestCase):
    def test_coverage(self):
        # Load reference coverage.
        reference_coverage = gpd.read_file(
            f"{Path(__file__).resolve().parent}/test_data/reference_coverage.geojson"
        ).iloc[0]["geometry"]

        # Calculate coverage.
        radar = Radar(
            id=585,
            point=Point(
                lat=47.36700085728634,
                lon=8.537724304199216,
                alt=407.83600886023686,
            ),
            power=20000,
            erp=800,
            antenna_height=10.0,
            diameter=2.0,
            frequency=1000.0,
            pulse_width=1,
            cpi_pulses=1,
            bandwidth=1,
            pfa=1e-6,
            min_elevation=-20.0,
            max_elevation=60.0,
            rotation_time=10.0,
            polarization=Polarization.HORIZONTAL,
        )

        start = Point(
            lat=radar.lat,
            lon=radar.lon,
            alt=elevationAt(radar.lat, radar.lon),
        )

        target_cross_section: float = 2.0
        target_flight_height = 1000.0

        max_dist = radar_eq_max_dist(
            radar,
            target_cross_section,
        )

        coverage = calculate_coverage(
            start, max_dist, target_flight_height, d_theta=0.1
        )

        # Convert to cartesian coordinates and calculate Dice simularity metric.
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        A = shapely.ops.transform(transformer.transform, reference_coverage)
        B = shapely.ops.transform(transformer.transform, coverage)

        dice = 2 * A.intersection(B).area / (A.area + B.area)

        self.assertGreater(dice, 0.989)


if __name__ == "__main__":
    unittest.main()
