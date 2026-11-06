import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import pyproj

from theia.export_kadas import KadasLayer
from theia.types import Point

BERN = Point(lat=46.948, lon=7.4474, alt=540.0)
THUN = Point(lat=46.758, lon=7.628, alt=560.0)


class KadasLayerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "export.gpkg")

    def tearDown(self):
        self._tmp.cleanup()

    def test_markers_only(self):
        layer = KadasLayer("pois")
        layer.add_marker(BERN)
        layer.add_marker(THUN)
        layer.export_layer(self.path)

        self.assertEqual([name for name, _ in pyogrio.list_layers(self.path)], ["pois"])
        gdf = gpd.read_file(self.path, layer="pois")
        self.assertEqual(gdf.crs.to_epsg(), 4326)
        self.assertEqual(len(gdf), 2)
        self.assertAlmostEqual(gdf.geometry[0].x, BERN.lon)
        self.assertAlmostEqual(gdf.geometry[0].y, BERN.lat)
        self.assertAlmostEqual(gdf["alt"][0], BERN.alt)
        self.assertAlmostEqual(gdf.geometry[1].x, THUN.lon)
        self.assertAlmostEqual(gdf.geometry[1].y, THUN.lat)
        self.assertAlmostEqual(gdf["alt"][1], THUN.alt)

    def test_marker_with_radius(self):
        layer = KadasLayer("radars")
        layer.add_marker(THUN)
        layer.add_marker_with_radius(BERN, 10_000.0)
        layer.export_layer(self.path)

        points = gpd.read_file(self.path, layer="radars")
        self.assertEqual(len(points), 2)
        self.assertTrue(np.isnan(points["radius_m"][0]))
        self.assertAlmostEqual(points["radius_m"][1], 10_000.0)

        ranges = gpd.read_file(self.path, layer="radars_ranges")
        self.assertEqual(len(ranges), 1)
        lons, lats = np.asarray(ranges.geometry[0].exterior.coords.xy)
        _, _, dist = pyproj.Geod(ellps="WGS84").inv(
            np.full_like(lons, BERN.lon), np.full_like(lats, BERN.lat), lons, lats
        )
        np.testing.assert_allclose(dist, 10_000.0, rtol=1e-6)

    def test_multiple_layers_in_one_file(self):
        a = KadasLayer("a")
        a.add_marker(BERN)
        a.export_layer(self.path)
        b = KadasLayer("b")
        b.add_marker(THUN)
        b.export_layer(self.path)

        names = {name for name, _ in pyogrio.list_layers(self.path)}
        self.assertEqual(names, {"a", "b"})

    def test_invalid_input(self):
        layer = KadasLayer("x")
        with self.assertRaises(ValueError):
            layer.export_layer(self.path)
        with self.assertRaises(ValueError):
            layer.add_marker_with_radius(BERN, 0.0)
        layer.add_marker(BERN)
        with self.assertRaises(ValueError):
            layer.export_layer(str(Path(self._tmp.name) / "export.shp"))


if __name__ == "__main__":
    unittest.main()
