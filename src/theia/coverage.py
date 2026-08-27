import functools
from multiprocessing import Pool

import numpy as np
import shapely
from scipy.ndimage import binary_dilation, binary_erosion

from theia.detection.pcl import PclDetector
from theia.distance import burstvincentydistance, line_of_sight_distance
from theia.grids import LatLonHeightGrid
from theia.terrain import AbstractTerrainModel
from theia.types import PclSensor, Point
from theia.util import mask_to_polygon


def calculate_coverage(
    terrain: AbstractTerrainModel,
    point: Point,
    d_max: float,
    target_alt: float,
    dlat: float = 0.01,
    dlon: float = 0.025,
):
    p = (point.lat, point.lon)
    north = burstvincentydistance(p, d_max, 0, target_alt)
    east = burstvincentydistance(p, d_max, 90, target_alt)
    south = burstvincentydistance(p, d_max, 180, target_alt)
    west = burstvincentydistance(p, d_max, 270, target_alt)
    grid = LatLonHeightGrid(
        lat_start=south.lat,
        lat_stop=north.lat + dlat,
        lat_res=dlat,
        lon_start=west.lon,
        lon_stop=east.lon + dlon,
        lon_res=dlon,
        height_start=target_alt,
        height_stop=target_alt,
        height_res=1,
    )

    return _calculate_monostatic_coverage(point, terrain, grid, d_max)


def _calculate_monostatic_coverage(
    point: Point,
    terrain: AbstractTerrainModel,
    grid: LatLonHeightGrid,
    d_max: float,
) -> np.ndarray:
    points = grid.points
    mask = np.empty((points.shape[0],), dtype=bool)
    for i, p in enumerate(points):
        # Distance check: We have a rectangular grid!
        mask[i] = line_of_sight_distance(
            point.lat,
            point.lon,
            point.alt,
            p[0],
            p[1],
            p[2],
        ) <= d_max and terrain.has_line_of_sight(
            point,
            Point(lat=p[0], lon=p[1], alt=p[2]),
        )
    mask = mask.reshape(grid.n_points[:2])
    mask = binary_dilation(binary_erosion(mask))

    return mask_to_polygon(
        mask,
        grid.lat_start - grid.lat_res,
        grid.lat_res,
        grid.lon_start - grid.lon_res,
        grid.lon_res,
    )


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
