from __future__ import annotations
from dataclasses import dataclass
import itertools
import threading
from typing import Callable
from zipfile import ZipFile

from joblib import Parallel, delayed
import numba
import numpy as np
from pydantic import ConfigDict

from theia.coordinates import CoordinateTransformations
import theia.terrain
from theia.terrain import AbstractTerrainModel, SrtmTerrainModel
from theia.types import Point


class FastSrtmModel(AbstractTerrainModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tree: HbvTree
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
        return self.tree.has_line_of_sight(ray)


def build_bounding_boxes(
    lats: np.ndarray, lons: np.ndarray, data: np.ndarray
) -> np.ndarray:
    """
    Build the leafs of the terrain HBV tree.

    Iterates through the terrain elevation grid in geodetic space and builds
    an AABB in ECEF space for each grid cell.

    Parameters
    ----------
    lats: np.ndarray
        Array of latitude values [°]; shape (Nlats,)
    lons: np.ndarray
        Array of longitude values [°]; shape (Nlons,)
    data: np.ndarray
        Elevation data [meters above sea level]; shape (Nlats, Nlons)

    Returns
    -------
    np.ndarray
        Array of shape (Nlats, Nlons, 6), where the last dimension contains
        the coordinates of the bounding box for each (lat, lon) cell in ECEF
        space.
        The coordinate order is:
        x_min, y_min, z_min, x_max, y_max, z_max
    """
    half_spacing_lat = 0.5 * (lats[1] - lats[0])
    half_spacing_lon = 0.5 * (lons[1] - lons[0])
    assert np.isclose(half_spacing_lat, half_spacing_lon)
    half_spacing = half_spacing_lat

    def process_row(i, lat):
        row = np.empty((len(lons), 6))
        for j, lon in enumerate(lons):
            alt = data[i, j]
            alt_below = alt - 30
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
            row[j, :3] = np.min(points, axis=0)
            row[j, 3:] = np.max(points, axis=0)
        return row

    results = Parallel(n_jobs=-1)(
        delayed(process_row)(i, lat) for i, lat in enumerate(lats)
    )

    bbox_coords = np.empty((len(lats), len(lons), 6))
    for i, row in enumerate(results):
        bbox_coords[i] = row
    return bbox_coords


BBoxParams = tuple[float, float, float, float, float, float]


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
        Examples: np.min, np.max, np.mean, np.sum.

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


class HbvTree:
    """
    Hierarchical bounding volumes tree for terrain data.

    The bounding volumes are axis-aligned bounding boxes (AABB) in Cartesian space.
    """

    # def __getitem__(
    #     self, index: int | tuple[int, int, int]
    # ) -> tuple[BBoxParams, tuple[int, int, int, int]]:
    #     """
    #     Access the node either by flat index or (depth, i, j) index.

    #     Returns
    #     -------
    #     BBoxParams:
    #         Parameters of the node with the given index
    #     tuple[int, int, int, int]:
    #         Flat indices of the four children;
    #         equal to (-1, -1, -1, -1) for lead nodes
    #     """
    #     if isinstance(index, int):
    #         flat_index = index
    #     elif isinstance(index) is tuple and len(index) == 3:
    #         depth, i, j = index
    #         flat_index = (
    #             self._cum_N_per_depth[depth - 1]
    #             + self._N_linear_per_depth[depth - 1] * i
    #             + j
    #         )
    #     else:
    #         raise RuntimeError()
    #     return self._data[flat_index, :], self._children[flat_index, :]

    def __init__(self, data: np.ndarray, children: np.ndarray):
        data = np.ascontiguousarray(data, dtype=np.float64)
        children = np.ascontiguousarray(children, dtype=np.int64)

        self._data = data
        """
        Parameters of each AABB, i. e. xmin, ymin, zmin, xmax, ymax, zmax.
        The nodes are ordered by ascending depth, i. e. larger nodes come first.
        Shape (N_nodes, 6)
        """
        self._children = children
        """
        Flat indices of the child nodes. Value -1 indicates no children.
        The nodes are ordered by ascending depth, i. e. larger nodes come first.
        Shape (N_nodes, 4). 
        """
        # Convert number of nodes to depth.
        max_depths = {
            1: 1,
            5: 2,
            21: 3,
            85: 4,
            341: 5,
            1365: 6,
            5461: 7,
            21845: 8,
            87381: 9,
            349525: 10,
            1398101: 11,
            5592405: 12,
            22369621: 13,
            89478485: 14,
            357913941: 15,
            1431655765: 16,
            5726623061: 17,
            22906492245: 18,
            91625968981: 19,
            366503875925: 20,
        }
        max_depth = max_depths[data.shape[0]]
        self._N_linear_per_depth = [2**i for i in range(max_depth + 1)]
        self._cum_N_per_depth = [0] + list(
            itertools.accumulate((2 ** (2 * i) for i in range(max_depth)))
        )

        # Maximum DFS stack occupancy for a complete quadtree:
        #   start with 1 (root), each level pops 1 and pushes 4 → net +3.
        #   After traversing d levels: 1 + 3*(d-1) + 4 = 3*d + 2.
        # A small constant over-allocation is cheaper than a bounds check.
        self._stack_size = 3 * max_depth + 4

        # Thread-local storage: numba releases the GIL so multiple Python
        # threads can call has_line_of_sight concurrently on the same tree.
        # Each thread gets its own stack buffer; no locks required.
        self._tls = threading.local()

    @staticmethod
    def from_leaf_bboxes(leaf_bbox_coordinates: np.ndarray) -> HbvTree:
        """
        Parameters
        ----------
        leaf_bbox_coordinates: np.ndarray
            Parameters of the AABB. Shape: ``(N, N, 6)``
            The first two axes refer to the geodetic coordinate grid on which
            the AABBs are defined. The last axis corresponds to the AABB parameters
                ``xmin, ymin, zmin, xmax, ymax, zmax``.
            ``N`` is the number of leafs.
        """
        max_depth = int(np.log2(leaf_bbox_coordinates.shape[0])) + 1
        all_bbox_coords: dict[int, np.ndarray] = {max_depth: leaf_bbox_coordinates}
        for depth in range(max_depth, 1, -1):
            all_bbox_coords[depth - 1] = build_higher_level(all_bbox_coords[depth])
        N_linear_per_depth = [2**i for i in range(max_depth + 1)]
        cum_N_per_depth = [0] + list(
            itertools.accumulate((2 ** (2 * i) for i in range(max_depth)))
        )
        N_nodes = cum_N_per_depth[max_depth]

        data: np.ndarray = np.empty((N_nodes, 6))
        children: np.ndarray = np.full((N_nodes, 4), -1, dtype=np.int64)
        for depth in range(1, max_depth + 1):
            N = N_linear_per_depth[depth - 1]
            N_deeper = N_linear_per_depth[depth]
            for i, j in itertools.product(range(N), range(N)):
                parent_index = (
                    cum_N_per_depth[depth - 1] + N_linear_per_depth[depth - 1] * i + j
                )
                data[parent_index] = all_bbox_coords[depth][i, j, :]

                if depth < max_depth:
                    index1 = cum_N_per_depth[depth] + i * N_deeper + j
                    index2 = cum_N_per_depth[depth] + i * N_deeper + j + 1
                    index3 = cum_N_per_depth[depth] + (i + 1) * N_deeper + j
                    index4 = cum_N_per_depth[depth] + (i + 1) * N_deeper + j + 1
                    children[parent_index] = (index1, index2, index3, index4)

        return HbvTree(data, children)

    def save(self, filename: str):
        with ZipFile(filename, "w") as zip_file:
            with zip_file.open("data.npy", "w") as file:
                np.save(file, self._data)
            with zip_file.open("children.npy", "w") as file:
                np.save(file, self._children)

    @staticmethod
    def load(filename: str) -> HbvTree:
        with ZipFile(filename, "r") as zip_file:
            with zip_file.open("data.npy", "r") as file:
                data = np.load(file)
            with zip_file.open("children.npy", "r") as file:
                children = np.load(file)
        return HbvTree(data=data, children=children)

    def _get_stack(self) -> np.ndarray:
        """Return this thread's scratch buffer, allocating on first access."""
        buf = getattr(self._tls, "stack", None)
        if buf is None:
            buf = np.empty(self._stack_size, dtype=np.int64)
            self._tls.stack = buf
        return buf

    def warmup(self) -> None:
        """
        Force JIT compilation synchronously.

        Call once after construction (or after loading from cache) to pay
        the ~1 s compilation cost at a predictable moment rather than on
        the first real query.  cache=True means subsequent process starts
        skip this entirely.
        """
        _los_kernel(
            np.zeros((1, 6), dtype=np.float64),
            np.full((1, 4), -1, dtype=np.int64),
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            np.empty(8, dtype=np.int64),
        )

    def has_line_of_sight(self, ray: Ray) -> bool:
        """True when no terrain blocks the ray between origin and t_max."""
        px, py, pz = ray.p_start
        dx, dy, dz = ray.direction
        return bool(
            _los_kernel(
                self._data,
                self._children,
                float(px),
                float(py),
                float(pz),
                1.0 / float(dx),
                1.0 / float(dy),
                1.0 / float(dz),
                float(ray.t_max),
                self._get_stack(),
            )
        )


# ---------------------------------------------------------------------------
# Module-level kernel — compiled once, reused for every HbvTree instance.
# Keeping it outside the class avoids recompilation per instance and lets
# numba see a stable type signature from the first call.
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _los_kernel(
    data: np.ndarray,
    children: np.ndarray,
    px: float,
    py: float,
    pz: float,
    idx: float,
    idy: float,
    idz: float,
    t_max: float,
    stack: np.ndarray,
) -> bool:
    """
    Iterative DFS.  Returns False as soon as a leaf AABB is intersected
    (terrain blocks the ray), True if the full tree is traversed with no hit.

    Axis-aligned rays are handled correctly: a zero direction component
    produces ±inf for the corresponding inv_d, which the slab test handles
    via IEEE-754 arithmetic without any special case.

    NaN (ray origin exactly on a slab face) is not guarded against; the
    comparison `ta > tb` returns False for NaN operands so the swap is
    skipped, and the subsequent `ta > t_enter` / `tb < t_exit` comparisons
    also return False — the axis contributes nothing to the interval, which
    is the conservative (non-pruning) choice.  False negatives are
    impossible; at worst one extra subtree is visited.

    Assumes a complete quadtree: every internal node has exactly 4 children.

    Parameters
    ----------
    data: np.ndarray
        Extents of all AABB. Shape: (N_nodes, 6), float64, C-contiguous
        The lower-depth (larger) AABB are placed first, the leaf nodes last.
    children: np.ndarray
        Children of all AABB nodes. Shape: (N_nodes, 4), int64, C-contiguous
        The lower-depth (larger) AABB are placed first, the leaf nodes last.
    px: float
        Ray starting point x coordinate
    py: float
        Ray starting point y coordinate
    pz: float
        Ray starting point z coordinate
    idx: float
        Inverted ray x-direction
    idy: float
        Inverted ray y-direction
    idz: float
        Inverted ray z-direction
    t_max: float
        Maximum distance along the ray
    stack: np.ndarray
        pre-allocated int64 scratch memory of length >= 3*depth+1
    """
    stack[0] = 0
    top = 1

    while top > 0:
        top -= 1
        node = stack[top]

        t_enter = 0.0
        t_exit = t_max

        # ---- X slab -------------------------------------------------------
        ta = (data[node, 0] - px) * idx
        tb = (data[node, 3] - px) * idx
        if ta > tb:
            ta, tb = tb, ta
        if ta > t_enter:
            t_enter = ta
        if tb < t_exit:
            t_exit = tb
        if t_enter > t_exit:
            continue  # early-out per axis

        # ---- Y slab -------------------------------------------------------
        ta = (data[node, 1] - py) * idy
        tb = (data[node, 4] - py) * idy
        if ta > tb:
            ta, tb = tb, ta
        if ta > t_enter:
            t_enter = ta
        if tb < t_exit:
            t_exit = tb
        if t_enter > t_exit:
            continue

        # ---- Z slab -------------------------------------------------------
        ta = (data[node, 2] - pz) * idz
        tb = (data[node, 5] - pz) * idz
        if ta > tb:
            ta, tb = tb, ta
        if ta > t_enter:
            t_enter = ta
        if tb < t_exit:
            t_exit = tb
        if t_enter > t_exit:
            continue

        # ---- leaf check ---------------------------------------------------
        if children[node, 0] == -1:
            return False  # terrain AABB hit → LOS blocked

        # ---- push all four children (unrolled for numba) ------------------
        stack[top] = children[node, 0]
        stack[top + 1] = children[node, 1]
        stack[top + 2] = children[node, 2]
        stack[top + 3] = children[node, 3]
        top += 4

    return True  # no leaf hit → clear LOS


def load_srtm_bboxes(
    lat_start: float,
    lat_stop: float,
    lon_start: float,
    lon_stop: float,
    subsample_factor: int = 1,
) -> np.ndarray:
    """
    Load SRTM data for the given region and stitch it together.
    Also apply padding so that ``N`` is a power of two.
    The padded elevation cells have altitude zero.

    Returns
    -------
    lats: np.ndarray
        Latitude values [°]; shape (N,)
    lons: np.ndarray
        Longitude values [°]; shape (N,)
    data: np.ndarray
        Altitude values [meters above sea level]; shape (N, N)
    """

    lat_start = np.floor(lat_start)
    lat_stop = np.ceil(lat_stop)
    lon_start = np.floor(lon_start)
    lon_stop = np.ceil(lon_stop)

    rows = []
    for lat in range(int(lat_stop) - 1, int(lat_start) - 1, -1):  # ← north→south
        row = []
        for lon in range(int(lon_start), int(lon_stop)):
            data = theia.terrain.load_hgt_file(lat, lon)[::-1, :]  # row 0 = north edge
            if lon != lon_stop - 1:
                data = data[:, :-1]
            if lat != lat_start:
                data = data[:-1, :]
            row.append(data)
        rows.append(np.hstack(row))
    data = np.vstack(rows)

    lats = np.linspace(lat_start, lat_stop, data.shape[0])
    lons = np.linspace(lon_start, lon_stop, data.shape[1])

    # Pad the data such that its shape is square with side length a power of 2.
    N = 2 ** int(np.ceil(np.log2(max(len(lats), len(lons)))))

    data_new = np.zeros((N, N))
    data_new[: data.shape[0], : data.shape[1]] = data
    lats_new = np.arange(N) * (lats[1] - lats[0]) + lats[0]
    lons_new = np.arange(N) * (lons[1] - lons[0]) + lons[0]

    assert np.allclose(lats_new[: len(lats)], lats)
    assert np.allclose(lons_new[: len(lons)], lons)
    assert np.allclose(data_new[: data.shape[0], : data.shape[1]], data)

    data = data_new
    lats = lats_new
    lons = lons_new

    assert np.log2(len(lats)).is_integer()
    assert np.log2(len(lons)).is_integer()
    assert len(lats) == len(lons)

    return lats, lons, data


def build_terrain_tree(
    lat_start: float,
    lat_stop: float,
    lon_start: float,
    lon_stop: float,
) -> HbvTree:
    lats, lons, data = load_srtm_bboxes(
        lat_start,
        lat_stop,
        lon_start,
        lon_stop,
    )
    bbox_coords = build_bounding_boxes(lats, lons, data)
    return HbvTree.from_leaf_bboxes(bbox_coords)
