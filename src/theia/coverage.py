import functools
import json
from multiprocessing import Pool
import tempfile
import numpy as np
import geopandas as gpd
import shapely
from theia.detection.pcl import PclDetector
from theia.distance import burstvincentydistance
from theia.grids import LatLonHeightGrid
from theia.line_of_sight import line_of_sight_along_ray
from theia.terrain import AbstractTerrainModel
from theia.types import PclSensor, Point


def calculate_coverage(
    terrain_model: AbstractTerrainModel,
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
        terrain_model=terrain_model,
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


def calculate_range_polygon(
    start: Point,
    max_dist: float,
    target_alt: float,
    d_theta: float = 0.5,
) -> shapely.geometry.polygon.Polygon:
    thetas = np.arange(0.0, 360.0, d_theta)

    visible_points = [
        burstvincentydistance(
            (start.lat, start.lon),
            max_dist,
            theta,
            target_alt,
        )
        for theta in thetas
    ]
    if visible_points[-1] != visible_points[0]:
        visible_points.append(visible_points[0])

    return shapely.geometry.polygon.Polygon([[p.lon, p.lat] for p in visible_points])


def minimum_detectable_rcs_grid_functional(
    detector: PclDetector, sensor: PclSensor, grid: LatLonHeightGrid
) -> np.ndarray:
    return detector.minimum_detectable_rcs_grid(
        sensor.receiver,
        sensor.transmitter,
        grid,
    )


def pcl_track_init_update_masks_parallel(
    detector: PclDetector,
    sensors: list[PclSensor],
    grid: LatLonHeightGrid,
    rcs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate binary masks for PCL track init (at least 3 sensors detect) and
    track update (at least 1 sensor detects).

    Parameters
    ----------
    detector: PclDetector
        Detector model to use
    sensors: list[PclSensor]
        PCL Sensors
    grid: LatLonHeightGrid
        Grid on which to calculate detectability
    rcs: float
        Radar cross section of the target to be detected.

    Returns
    -------
    track_init_mask: np.ndarray
        Binary mask indicating where on the grid a track can be initialized using
        only the PCL sensors
    track_update_mask: np.ndarray
        Binary mask indicating where on the grid a track can be updated using
        only the PCL sensors
    """
    with Pool(5) as p:
        worker = functools.partial(
            minimum_detectable_rcs_grid_functional,
            detector,
            grid=grid,
        )
        min_detectable_rcs_grids = p.map(worker, sensors)

    detection_grids = [rcs_grid <= rcs for rcs_grid in min_detectable_rcs_grids]
    n_detections_grid = sum(detection_grids)
    track_update_mask = np.logical_and(0 < n_detections_grid, n_detections_grid < 3)
    track_init_mask = 3 <= n_detections_grid

    return track_init_mask, track_update_mask
