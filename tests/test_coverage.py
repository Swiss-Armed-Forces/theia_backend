from pathlib import Path
import unittest

import geopandas as gpd
from pyproj import Transformer
import shapely

from theia.coverage import calculate_coverage
from theia.terrain import elevationAt
from theia.radar_equation import radar_eq_max_dist
from theia.types import Point, Polarization, Radar, Receiver, Transmitter


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
        ah = 10.0
        bandwidth = 1
        diameter = 2.
        radar = Radar(
            transmitter=Transmitter(
                id=585,
                point=p,
                power=20000,
                erp=800,
                antenna_height=ah,
                antenna_diameter=diameter,
                frequency=1000.0,
                pulse_width=1,
                bandwidth=bandwidth,
                polarization=Polarization.HORIZONTAL,
            ),
            receiver=Receiver(
                id=585,
                point=p,
                antenna_height=ah,
                diameter=2.0,
                cpi_pulses=1,
                pfa=1e-6,
                min_elevation=-20.0,
                max_elevation=60.0,
                rotation_time=10.0,
                bandwidth=bandwidth,
            ),
        )

        start = Point(
            lat=p.lat,
            lon=p.lon,
            alt=elevationAt(p.lat, p.lon),
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
