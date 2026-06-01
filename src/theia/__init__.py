from theia.terrain import SrtmTerrainModel
from theia.terrain_fast_los import FastSrtmModel

from .types import Point, AbstractSensor
from .radar_equation import calculate_maximum_monostatic_range
from .distance import haversine, linspace
from .coverage import calculate_coverage


TerrainModel = SrtmTerrainModel | FastSrtmModel
