from importlib.metadata import version

from theia.terrain import DummyTerrain, FillTerrainModel, SrtmTerrainModel
from theia.terrain_fast_los import FastSrtmModel

__version__ = version("theia")

TerrainModel = SrtmTerrainModel | FastSrtmModel | DummyTerrain | FillTerrainModel
