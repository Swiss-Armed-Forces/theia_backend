import unittest
from pathlib import Path

import geopandas as gpd
import shapely
from pyproj import Transformer

from theia.coverage import calculate_coverage
from theia.terrain import SrtmTerrainModel
from theia.types import Point


class CoverageTest(unittest.TestCase):
    def test_coverage(self):
        terrain_model = SrtmTerrainModel()
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
            alt=terrain_model.elevationAt(p.lat, p.lon),
        )

        target_flight_height = 1000.0

        # From openBURST.
        max_dist = 16500.001

        coverages = calculate_coverage(
            terrain_model,
            start,
            max_dist,
            target_flight_height,
            0.001,
            0.001,
        )
        self.assertTrue(len(coverages) == 1)
        coverage = coverages[0]

        # Convert to cartesian coordinates and calculate Dice simularity metric.
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        A = shapely.ops.transform(transformer.transform, reference_coverage)
        B = shapely.ops.transform(transformer.transform, coverage)

        dice = 2 * A.intersection(B).area / (A.area + B.area)

        self.assertGreater(dice, 0.989)


if __name__ == "__main__":
    unittest.main()
