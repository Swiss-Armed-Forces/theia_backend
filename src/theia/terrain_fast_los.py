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
from tqdm import tqdm

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
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    n_jobs: int = -1,
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
    n_jobs: int, default -1
        Number of parallel jobs. By default, joblib will try to use all CPUs

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

    results = list(
        tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(process_row)(i, lat) for i, lat in enumerate(lats)
            ),
            total=len(lats),
        )
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

    def __post_init__(self):
        if np.allclose(self.direction, 0):
            raise ValueError("Ray direction must not be zero!")


def _num_nodes_to_depth(N: int) -> int:
    """
    Convert number of nodes in a quadtree to its depth.
    """
    # The number of nodes in a tree of depth d is N = 4^0 + 4^1 + 4^2 + ... + 4^(d-1),
    # which is a geometric series and therefore simplifies to
    # N = (4^d - 1) / (4 - 1)
    # Solving for d yields:
    # d = 0.5 * log2(3N + 1)
    return int(np.round(0.5 * np.log2(3 * N + 1)))


def _depth_to_num_nodes(depth: int) -> int:
    """
    Get the number of nodes in a complete quadtree from its depth.
    """
    # The number of nodes in a tree of depth d is N = 4^0 + 4^1 + 4^2 + ... + 4^d,
    # which is a geometric series and therefore simplifies to
    # N = (4^d - 1) / (4 - 1)
    return int(np.round((4**depth - 1) / 3))


class HbvTree:
    """
    Hierarchical Bounding Volume (HBV) tree for terrain line-of-sight queries.

    The tree is a complete quadtree in which each node stores an
    axis-aligned bounding box (AABB) in Cartesian ECEF (x, y, z) space.  Internal
    nodes contain the tightest AABB that encloses all four children; leaf nodes
    correspond 1-to-1 with the input terrain grid cells.

    Nodes are stored in breadth-first order so that the root is at index 0 and
    leaves occupy the highest indices.  Within each depth level nodes are in
    row-major (C) order over the 2-D geodetic grid.

    Parameters
    ----------
    data:
        ``float64`` array, shape ``(N_nodes, 6)``.  Each row contains
        ``[xmin, ymin, zmin, xmax, ymax, zmax]`` for the corresponding AABB.
    children:
        ``int64`` array, shape ``(N_nodes, 4)``.  Each row contains the flat
        indices of the four child nodes.  ``-1`` means "no child" (leaf).

    Thread safety
    -------------
    :meth:`has_line_of_sight` is thread-safe.  The Numba kernel releases the
    GIL; each calling thread uses its own scratch stack buffer allocated via
    :attr:`_tls` (thread-local storage) on first use.

    See Also
    --------
    HbvTree.from_leaf_bboxes : Recommended constructor from a leaf-AABB grid.
    HbvTree.load             : Load a previously saved tree from disk.
    """

    def __init__(self, data: np.ndarray, children: np.ndarray):
        data = np.ascontiguousarray(data, dtype=np.float64)
        children = np.ascontiguousarray(children, dtype=np.int64)

        self._data = data
        """
        AABB parameters for every node: ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
        Nodes are in breadth-first / ascending-depth order (root at index 0).
        Shape ``(N_nodes, 6)``.
        """

        self._children = children
        """
        Flat child indices for every node.  ``-1`` indicates no child (leaf).
        Nodes are in breadth-first / ascending-depth order (root at index 0).
        Shape ``(N_nodes, 4)``.
        """

        # Stack size for iterative DFS over a complete quadtree:
        #   Depth d → at most 1 + 3*(d-1) + 4 = 3*d + 2 entries on the stack
        #   simultaneously.  A small constant over-allocation avoids
        #   per-push bounds checks inside the hot kernel.
        max_depth = _num_nodes_to_depth(data.shape[0])
        self._stack_size = 3 * max_depth + 4

        # Thread-local scratch buffers; allocated lazily on first use per thread.
        self._tls = threading.local()

    @staticmethod
    def from_leaf_bboxes(leaf_bbox_coordinates: np.ndarray) -> HbvTree:
        """
        Build a complete quadtree from a 2-D grid of leaf AABBs.

        The input grid must be square with a side length that is a power of
        two.  The tree is constructed bottom-up: each internal node's AABB is
        the union of its four children's AABBs.

        Parameters
        ----------
        leaf_bbox_coordinates:
            ``float64`` array, shape ``(N, N, 6)``.
            The first two axes index the geodetic grid; the last axis contains
            ``[xmin, ymin, zmin, xmax, ymax, zmax]`` for each leaf cell.
            ``N`` must be a positive power of two.

        Returns
        -------
        HbvTree
            Fully constructed tree ready for line-of-sight queries.

        Raises
        ------
        ValueError
            If the grid side length is not a positive power of two.

        Examples
        --------
        >>> import numpy as np
        >>> leaves = np.zeros((2, 2, 6))          # 2×2 grid of unit AABBs
        >>> leaves[..., 3:] = 1.0                  # xmax=ymax=zmax=1
        >>> tree = HbvTree.from_leaf_bboxes(leaves)
        """
        N = leaf_bbox_coordinates.shape[0]
        if N == 0 or (N & (N - 1)) != 0:
            raise ValueError(
                f"Leaf grid side length must be a positive power of two; got {N}."
            )

        max_depth = int(np.round(np.log2(N))) + 1
        all_bbox_coords: dict[int, np.ndarray] = {max_depth: leaf_bbox_coordinates}
        for depth in range(max_depth, 1, -1):
            all_bbox_coords[depth - 1] = build_higher_level(all_bbox_coords[depth])

        # depth_start_index[d] = flat index of the first node at depth d.
        # Nodes at depth d occupy positions [total_nodes_in_(d-1), total_nodes_in_d).
        depth_start_index = [_depth_to_num_nodes(d - 1) for d in range(max_depth + 2)]
        N_nodes = _depth_to_num_nodes(max_depth)

        data: np.ndarray = np.empty((N_nodes, 6))
        children: np.ndarray = np.full((N_nodes, 4), -1, dtype=np.int64)

        for depth in range(1, max_depth + 1):
            # Number of nodes along one side at this depth.
            n_side = 2 ** (depth - 1)
            n_side_deeper = 2 * n_side
            for i, j in itertools.product(range(n_side), range(n_side)):
                parent_index = depth_start_index[depth] + n_side * i + j
                data[parent_index] = all_bbox_coords[depth][i, j, :]

                if depth < max_depth:
                    base = depth_start_index[depth + 1]
                    # The four children of (i, j) at depth+1 form a 2×2 block
                    # starting at grid position (2i, 2j).
                    index1 = base + (2 * i) * n_side_deeper + (2 * j)  # (2i,   2j  )
                    index2 = (
                        base + (2 * i) * n_side_deeper + (2 * j) + 1
                    )  # (2i,   2j+1)
                    index3 = (
                        base + (2 * i + 1) * n_side_deeper + (2 * j)
                    )  # (2i+1, 2j  )
                    index4 = (
                        base + (2 * i + 1) * n_side_deeper + (2 * j) + 1
                    )  # (2i+1, 2j+1)
                    children[parent_index] = (index1, index2, index3, index4)

        return HbvTree(data, children)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str):
        """
        Persist the tree to a ZIP archive containing two ``.npy`` files.

        Parameters
        ----------
        filename:
            Path to the output ``.zip`` file.  Existing files are overwritten.

        See Also
        --------
        HbvTree.load : Counterpart for loading a saved tree.
        """
        with ZipFile(filename, "w") as zip_file:
            with zip_file.open("data.npy", "w") as file:
                np.save(file, self._data)
            with zip_file.open("children.npy", "w") as file:
                np.save(file, self._children)

    @staticmethod
    def load(filename: str) -> HbvTree:
        """
        Load a tree that was previously saved with :meth:`save`.

        Parameters
        ----------
        filename:
            Path to a ``.zip`` archive produced by :meth:`save`.

        Returns
        -------
        HbvTree
            The reconstructed tree (without recomputing AABB merges).
        """
        with ZipFile(filename, "r") as zip_file:
            with zip_file.open("data.npy", "r") as file:
                data = np.load(file)
            with zip_file.open("children.npy", "r") as file:
                children = np.load(file)
        return HbvTree(data=data, children=children)

    def _get_stack(self) -> np.ndarray:
        """Return this thread's scratch buffer, allocating it on first access."""
        buf = getattr(self._tls, "stack", None)
        if buf is None:
            buf = np.empty(self._stack_size, dtype=np.int64)
            self._tls.stack = buf
        return buf

    def warmup(self) -> None:
        """
        Trigger Numba JIT compilation synchronously.

        Calling this once after construction (or after loading from cache)
        pays the ~1 s compilation cost at a predictable moment rather than
        on the first real query.  With ``cache=True`` on the kernel,
        subsequent process starts will skip compilation entirely.
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
        """
        Test whether a ray is unobstructed by any terrain leaf AABB.

        Parameters
        ----------
        ray:
            The query ray.  See :class:`Ray` for details.

        Returns
        -------
        bool
            ``True``  - the ray does **not** intersect any leaf AABB within
            ``[0, t_max]``; line of sight is clear.
            ``False`` - at least one leaf AABB is hit; terrain blocks the ray.

        Notes
        -----
        This method is thread-safe: the Numba kernel releases the GIL, and
        each thread allocates its own DFS (depth-first-search) stack buffer
        on first use.
        """
        px, py, pz = ray.p_start
        dx, dy, dz = ray.direction

        # Use the IEEE 754 reciprocal convention: 1/0 → ±inf.
        # The slab test handles ±inf correctly (parallel rays never enter a slab
        # whose normal is aligned with the zero-component axis, so t_enter > t_exit).
        def _safe_inv(v: float) -> float:
            return float("inf") if v == 0.0 else 1.0 / v

        return bool(
            _los_kernel(
                self._data,
                self._children,
                float(px),
                float(py),
                float(pz),
                _safe_inv(float(dx)),
                _safe_inv(float(dy)),
                _safe_inv(float(dz)),
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
