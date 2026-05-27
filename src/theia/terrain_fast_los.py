from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable

import numba
import numpy as np
import tqdm

from theia.coordinates import CoordinateTransformations
from theia.terrain import AbstractTerrainModel, SrtmTerrainModel
from theia.types import Point


class FastSrtmModel(AbstractTerrainModel):
    node: Node
    srtm_model: SrtmTerrainModel

    def elevationAt(self, lat: float, lon: float):
        self.srtm_model.elevationAt(lat, lon)

    def has_line_of_sight(self, p1: Point, p2: Point) -> bool:
        """
        Checks line of sight between two points accounting for Earth curvature
        and terrain elevation.

        step_m controls sampling resolution along the path.
        """
        p_start = CoordinateTransformations.geodetic_to_cartesian(
            p1.lat,
            p1.lon,
            p1.alt,
        )
        p1_ecef = np.array(p_start)
        p2_ecef = np.array(
            CoordinateTransformations.geodetic_to_cartesian(
                p2.lat,
                p2.lon,
                p2.alt,
            )
        )
        direction = p2_ecef - p1_ecef
        norm = np.linalg.norm(direction)
        direction = direction / norm
        ray = Ray(p_start=p1_ecef, direction=tuple(direction), t_max=norm)
        return self.node.intersect(ray) is None


@dataclass
class Ray:
    p_start: tuple[float, float, float]
    """
    Start point of a ray in ECEF coordinates [m]
    """
    direction: tuple[float, float, float]
    """
    Unit direction of the ray in ECEF coordinates [m]
    """
    t_max: float
    """
    Maximum ray parameter to consider (corresponds to maximum distance, but is unitless)
    """


@numba.njit
def _intersect_current_level(
    bounds: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ],
) -> float | None:
    """
    Proper AABB slab test.

    For each axis compute the entry/exit t values of the ray against that
    axis-aligned slab.  The ray hits the box only when all three entry
    intervals overlap, i.e. max(t_enters) <= min(t_exits) and the overlap
    is not entirely behind the ray origin.

    Returns the t of the *first surface hit*:
        - t_enter  when the ray starts outside the box (t_enter >= 0)
        - t_exit   when the ray starts inside  the box (t_enter <  0)
    Returns None on a miss.
    """
    t_enter = -math.inf
    t_exit = math.inf

    for minimum, maximum, origin, d in bounds:
        if d == 0:
            # Ray is parallel to this slab; miss if origin is outside it.
            if origin < minimum or origin > maximum:
                return None
        else:
            t1 = (minimum - origin) / d
            t2 = (maximum - origin) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)

    # No overlap between slabs, or the whole overlap is behind the ray.
    if t_enter > t_exit or t_exit < 0:
        return None

    return t_enter if t_enter >= 0 else t_exit


@dataclass
class Node:
    """
    Axis-aligned bounding box (AABB) representing the terrain in ECEF coordinates.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    children: tuple[Node, Node, Node, Node] | None

    def __str__(self) -> str:
        return f"Node(x_min={self.x_min}, x_max={self.x_max}, y_min={self.y_min}, y_max={self.y_max}, z_min={self.z_min}, z_max={self.z_max})"

    def is_leaf(self) -> bool:
        return self.children is None

    def depth(self) -> int:
        return 1 if self.children is None else self.children[0].depth() + 1

    def _intersect_current_level(self, ray: Ray) -> float | None:
        """
        Proper AABB slab test.

        For each axis compute the entry/exit t values of the ray against that
        axis-aligned slab.  The ray hits the box only when all three entry
        intervals overlap, i.e. max(t_enters) <= min(t_exits) and the overlap
        is not entirely behind the ray origin.

        Returns the t of the *first surface hit*:
          - t_enter  when the ray starts outside the box (t_enter >= 0)
          - t_exit   when the ray starts inside  the box (t_enter <  0)
        Returns None on a miss.
        """
        bounds = (
            (self.x_min, self.x_max, ray.p_start[0], ray.direction[0]),
            (self.y_min, self.y_max, ray.p_start[1], ray.direction[1]),
            (self.z_min, self.z_max, ray.p_start[2], ray.direction[2]),
        )
        return _intersect_current_level(bounds)

    def intersect(self, ray: Ray) -> float | None:
        """
        Intersect the AABB with a ray.

        Parameters
        ----------
        ray: Ray
            The ray to be checked for intersection

        Returns
        -------
        float | None
            Return the ray parameter of the first intersection with the AABB
            or None if there is no intersection.
        """
        intersection = self._intersect_current_level(ray)
        if intersection is None:
            return None
        if self.children is None:
            return intersection

        # Check for intersections at the lower level.
        intersections = (
            self.children[0].intersect(ray),
            self.children[1].intersect(ray),
            self.children[2].intersect(ray),
            self.children[3].intersect(ray),
        )
        intersection: float | None = None
        if intersections[0] is not None:
            intersection = intersections[0]
        if intersections[1] is not None:
            if intersection is None or intersections[1] < intersection:
                intersection = intersections[1]
        if intersections[2] is not None:
            if intersection is None or intersections[2] < intersection:
                intersection = intersections[2]
        if intersections[3] is not None:
            if intersection is None or intersections[3] < intersection:
                intersection = intersections[3]

        return intersection


def pool2d(
    arr: np.ndarray,
    operation: Callable[[np.ndarray, tuple[int, ...]], np.ndarray],
) -> np.ndarray:
    """
    Reduce a 2D array by applying an operation over non-overlapping 2x2 patches,
    producing an output array of half the side lengths.

    Parameters
    ----------
    arr : np.ndarray
        2D input array of shape (H, W).
    operation : callable
        Reduction function with signature f(arr, axis) -> np.ndarray.
        Defaults to np.min. Other examples: np.max, np.mean, np.sum.

    Returns
    -------
    np.ndarray
        2D array of shape (H//2, W//2).
    """
    # 1. Slice to the nearest even dimensions ([:h//2*2, :w//2*2])
    #    Odd-sized arrays are handled gracefully by dropping the last row/column.
    # 2. Reshape (H, W) → (H//2, 2, W//2, 2)
    #    This groups each axis into "block index" and "position within block".
    # 3. Reduce with .min(axis=(1, 3)) collapses the two within-block axes,
    #    leaving shape (H//2, W//2).
    h, w = arr.shape
    reshaped = arr[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2)
    return operation(reshaped, axis=(1, 3))


def build_higher_level(bbox_coords: np.ndarray) -> np.ndarray:
    x_min = pool2d(bbox_coords[:, :, 0], np.min)
    y_min = pool2d(bbox_coords[:, :, 1], np.min)
    z_min = pool2d(bbox_coords[:, :, 2], np.min)
    x_max = pool2d(bbox_coords[:, :, 3], np.max)
    y_max = pool2d(bbox_coords[:, :, 4], np.max)
    z_max = pool2d(bbox_coords[:, :, 5], np.max)

    return np.stack((x_min, y_min, z_min, x_max, y_max, z_max), axis=-1)


def build_tree(bbox_coords: np.ndarray) -> Node:
    max_depth = int(np.log2(bbox_coords.shape[0])) + 1
    all_bbox_coords: dict[int, np.ndarray] = {max_depth: bbox_coords}
    for depth in range(max_depth, 1, -1):
        all_bbox_coords[depth - 1] = build_higher_level(all_bbox_coords[depth])
    N_per_depth = {depth: coords.shape[0] for depth, coords in all_bbox_coords.items()}

    # Build the nodes.
    all_nodes: dict[int, dict[tuple[int, int], Node]] = {}
    for depth in range(max_depth, 0, -1):
        coords = all_bbox_coords[depth]
        all_nodes[depth] = {}
        for i in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                all_nodes[depth][(i, j)] = Node(
                    x_min=coords[i, j, 0],
                    y_min=coords[i, j, 1],
                    z_min=coords[i, j, 2],
                    x_max=coords[i, j, 3],
                    y_max=coords[i, j, 4],
                    z_max=coords[i, j, 5],
                    children=None,
                )

    # Link the nodes.
    for depth in range(1, max_depth):
        for i in range(N_per_depth[depth]):
            for j in range(N_per_depth[depth]):
                all_nodes[depth][(i, j)].children = (
                    all_nodes[depth + 1][(2 * i, 2 * j)],
                    all_nodes[depth + 1][(2 * i, 2 * j + 1)],
                    all_nodes[depth + 1][(2 * i + 1, 2 * j)],
                    all_nodes[depth + 1][(2 * i + 1, 2 * j + 1)],
                )
    assert len(all_nodes[1]) == 1
    return all_nodes[1][(0, 0)]


def build_bounding_boxes(
    lats: np.ndarray, lons: np.ndarray, data: np.ndarray
) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
        Array of shape (len(lats), len(lons), 6), where the last dimension contains
        the coordinates of the bounding box for each (lat, lon) cell in ECEF
        space.
        The coordinate order is:
        x_min, y_min, z_min, x_max, y_max, z_max
    """
    half_spacing_lat = 0.5 * (lats[1] - lats[0])
    half_spacing_lon = 0.5 * (lons[1] - lons[0])
    assert np.isclose(half_spacing_lat, half_spacing_lon)
    half_spacing = half_spacing_lat

    bbox_coords = np.empty((len(lats), len(lons), 6))
    for i, lat in tqdm.tqdm(enumerate(lats), total=len(lats)):
        for j, lon in enumerate(lons):
            alt = data[i, j]
            alt_below = np.clip(alt, 0, None) - 30
            p1 = CoordinateTransformations.geodetic_to_cartesian(
                lat - half_spacing,
                lon - half_spacing,
                alt,
            )
            p2 = CoordinateTransformations.geodetic_to_cartesian(
                lat + half_spacing,
                lon - half_spacing,
                alt,
            )
            p3 = CoordinateTransformations.geodetic_to_cartesian(
                lat - half_spacing,
                lon + half_spacing,
                alt,
            )
            p4 = CoordinateTransformations.geodetic_to_cartesian(
                lat + half_spacing,
                lon + half_spacing,
                alt,
            )
            p5 = CoordinateTransformations.geodetic_to_cartesian(
                lat - half_spacing,
                lon - half_spacing,
                alt_below,
            )
            p6 = CoordinateTransformations.geodetic_to_cartesian(
                lat + half_spacing,
                lon - half_spacing,
                alt_below,
            )
            p7 = CoordinateTransformations.geodetic_to_cartesian(
                lat - half_spacing,
                lon + half_spacing,
                alt_below,
            )
            p8 = CoordinateTransformations.geodetic_to_cartesian(
                lat + half_spacing,
                lon + half_spacing,
                alt_below,
            )

            points = np.stack([p1, p2, p3, p4, p5, p6, p7, p8])
            bbox_coords[i, j, :3] = np.min(points, axis=0)
            bbox_coords[i, j, 3:] = np.max(points, axis=0)
    return bbox_coords
