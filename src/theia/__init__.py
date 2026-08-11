from importlib.metadata import version

from theia.terrain import DummyTerrain, SrtmTerrainModel
from theia.terrain_fast_los import FastSrtmModel

from .coverage import calculate_coverage
from .distance import haversine, linspace
from .radar_equation import calculate_maximum_monostatic_range
from .types import AbstractSensor, Point

__version__ = version("theia")

TerrainModel = SrtmTerrainModel | FastSrtmModel | DummyTerrain
