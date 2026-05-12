from pathlib import Path
import unittest

import geopandas as gpd
from pyproj import Transformer
import shapely

from theia.coverage import calculate_coverage
from theia.terrain import SrtmTerrainModel
from theia.types import Point


class CoverageTest(unittest.TestCase):
    def test_coverage(self):
        # Load reference coverage.
        reference_coverage = gpd.read_file(
            f"{Path(__file__).resolve().parent}/test_data/reference_coverage.geojson"
        ).iloc[0]["geometry"]

        # Calculate coverage.
        p = Point(
            lat=47.36700085728634,
            lon=8.537724304199216,
            alt=407.83600886023686,
        )

        start = Point(
            lat=p.lat,
            lon=p.lon,
            alt=SrtmTerrainModel().elevationAt(p.lat, p.lon),
        )

        target_flight_height = 1000.0

        # From openBURST.
        max_dist = 16500.001

        coverage = calculate_coverage(
            SrtmTerrainModel(),
            start,
            max_dist,
            target_flight_height,
            d_theta=0.1,
        )

        # Convert to cartesian coordinates and calculate Dice simularity metric.
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        A = shapely.ops.transform(transformer.transform, reference_coverage)
        B = shapely.ops.transform(transformer.transform, coverage)

        dice = 2 * A.intersection(B).area / (A.area + B.area)

        self.assertGreater(dice, 0.989)


if __name__ == "__main__":
    unittest.main()
