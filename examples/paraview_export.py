import os

from theia.coordinates import POSITIONS_OF_INTEREST
from theia.export_paraview import ParaviewExporter, PointOfInterest
from theia.grids import LatLonHeightGrid
from theia.terrain import elevationAt
from theia.test_data import build_single_target_from_Bodensee

out_dir = "output/paraview"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

traj = build_single_target_from_Bodensee(alt=4000)

grid = LatLonHeightGrid(
    lat_start=46.28243,
    lat_stop=48.0046,
    lat_res=0.003,
    lon_start=8.0854,
    lon_stop=9.4973,
    lon_res=0.003,
    height_start=3000,
    height_stop=3000,
    height_res=1,
)

exporter = ParaviewExporter(
    out_dir,
    grid.lat_start,
    grid.lat_stop,
    grid.lat_res,
    grid.lon_start,
    grid.lon_stop,
    grid.lon_res,
    elevation_factor=5.0,
)

pois = [
    PointOfInterest(
        id=0,
        label="Rx",
        type="Rx",
        lat=46.9942,
        lon=8.5349,
        alt=elevationAt(46.9942, 8.5349) + 10,
    ),
    PointOfInterest(
        id=1,
        label="Rx Stallikon",
        type="Rx",
        lat=47.3258,
        lon=8.4914,
        alt=elevationAt(47.3258, 8.4914) + 10,
    ),
    PointOfInterest(
        id=2,
        label="Rx Uetliberg",
        type="Rx",
        lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
        lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
        alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
    ),
    PointOfInterest(
        id=3,
        label="Rx ZH Airport",
        type="Rx",
        lat=47.4629,
        lon=8.5693,
        alt=elevationAt(47.4629, 8.5693),
    ),
]

exporter.export([traj], pois)
