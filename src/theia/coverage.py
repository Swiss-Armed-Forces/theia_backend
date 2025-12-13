import functools
import json
from multiprocessing import Pool
import tempfile
import numpy as np
import geopandas as gpd
import shapely
from theia.line_of_sight import line_of_sight_along_ray
from theia.types import Point


def calculate_coverage(
    start: Point,
    max_dist: float,
    target_alt: float,
    dist_res: float = 30.0,
    d_theta: float = 0.5,
) -> shapely.geometry.polygon.Polygon:
    thetas = np.arange(0.0, 360.0, d_theta)
    f = functools.partial(
        line_of_sight_along_ray,
        start,
        target_alt=target_alt,
        d_max=max_dist,
        dist_res=dist_res,
    )
    with Pool(5) as p:
        visible_points = p.map(f, thetas)

    if visible_points[-1] != visible_points[0]:
        visible_points.append(visible_points[0])

    geojson = {
        "type": "Polygon",
        "coordinates": [[[p.lon, p.lat] for p in visible_points]],
    }

    with tempfile.NamedTemporaryFile("w", delete_on_close=False) as fp:
        json.dump(geojson, fp)
        fp.close()
        with open(fp.name, "r") as file:
            return gpd.read_file(file).iloc[0]["geometry"]