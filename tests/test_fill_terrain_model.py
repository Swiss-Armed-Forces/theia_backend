import errno
import tempfile
import unittest
from unittest.mock import patch

import theia.terrain
from theia.terrain import (
    AbstractTerrainModel,
    DummyTerrain,
    FillTerrainModel,
    SrtmTerrainModel,
)
from theia.types import Point


class MissingTileTerrain(AbstractTerrainModel):
    """Terrain whose elevation data is missing north of ``lat_max``."""

    lat_max: float
    height: float

    def elevationAt(self, lat: float, lon: float) -> float:
        if lat > self.lat_max:
            raise FileNotFoundError(
                errno.ENOENT, "No such file or directory", f"tile_{int(lat)}.hgt"
            )
        return self.height


class FillTerrainModelLosTest(unittest.TestCase):
    def setUp(self):
        self.p1 = Point(lat=47.0, lon=8.0, alt=100.0)
        self.p2 = Point(lat=47.1, lon=8.0, alt=100.0)

    def test_delegates_to_child(self):
        # DummyTerrain's elevation (1 m) never blocks the ray, so the result
        # can only come from the child's own LOS check.
        terrain = FillTerrainModel(child=DummyTerrain(has_los=False), fill_value=0)
        self.assertFalse(terrain.has_line_of_sight(self.p1, self.p2))

    def test_missing_tile_uses_fill_value(self):
        # The ray leaves the data coverage; the fill value (0 m) does not
        # block it.
        child = MissingTileTerrain(lat_max=47.05, height=0.0)
        terrain = FillTerrainModel(child=child, fill_value=0)
        self.assertTrue(terrain.has_line_of_sight(self.p1, self.p2))

    def test_missing_tile_fill_value_blocks(self):
        child = MissingTileTerrain(lat_max=47.05, height=0.0)
        terrain = FillTerrainModel(child=child, fill_value=500)
        self.assertFalse(terrain.has_line_of_sight(self.p1, self.p2))

    def test_missing_tile_terrain_before_gap_blocks(self):
        # Terrain in the covered part blocks the ray even though the child's
        # check fails further along.
        child = MissingTileTerrain(lat_max=47.05, height=500.0)
        terrain = FillTerrainModel(child=child, fill_value=0)
        self.assertFalse(terrain.has_line_of_sight(self.p1, self.p2))


class FillTerrainModelMissingDataTest(unittest.TestCase):
    def setUp(self):
        theia.terrain.load_hgt_file.cache_clear()
        FillTerrainModel._warned_missing_files.clear()
        self.terrain = FillTerrainModel(child=SrtmTerrainModel(), fill_value=7)

    def tearDown(self):
        theia.terrain.load_hgt_file.cache_clear()
        FillTerrainModel._warned_missing_files.clear()

    def test_missing_tile_is_filled_with_one_warning(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with patch("theia.terrain.ELEVATION_DATA_DIR", empty_dir):
                with self.assertLogs(level="WARNING") as logs:
                    self.assertEqual(self.terrain.elevationAt(47.1, 8.1), 7)
                    self.assertEqual(self.terrain.elevationAt(47.2, 8.2), 7)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("N47E008.hgt", logs.output[0])

    def test_unset_data_dir_raises(self):
        with patch("theia.terrain.ELEVATION_DATA_DIR", None):
            with self.assertRaisesRegex(RuntimeError, "THEIA_ELEVATION_DATA_DIR"):
                self.terrain.elevationAt(47.1, 8.1)
            with self.assertRaisesRegex(RuntimeError, "THEIA_ELEVATION_DATA_DIR"):
                SrtmTerrainModel().covered_region()

    def test_nonexistent_data_dir_raises(self):
        with patch("theia.terrain.ELEVATION_DATA_DIR", "/nonexistent/hgt"):
            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                self.terrain.elevationAt(47.1, 8.1)
            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                self.terrain.has_line_of_sight(
                    Point(lat=47.0, lon=8.0, alt=100.0),
                    Point(lat=47.1, lon=8.0, alt=100.0),
                )


if __name__ == "__main__":
    unittest.main()
