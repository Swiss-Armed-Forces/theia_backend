import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from theia.config import TERRAIN_HBV_DATA_DIR
from theia.terrain import SrtmTerrainModel
from theia.terrain_fast_los import (
    FastSrtmModel,
    HbvTree,
    Ray,
    _depth_to_num_nodes,
    _los_kernel,
    _num_nodes_to_depth,
    build_bounding_boxes,
    build_higher_level,
    load_srtm_bboxes,
)
from theia.types import Point


def _make_stack(max_depth: int) -> np.ndarray:
    """Return a stack buffer sized for the given tree depth."""
    # docstring requires len >= 3*depth+1; +4 gives comfortable headroom.
    return np.empty(3 * max_depth + 4, dtype=np.int64)


def _single_leaf(box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)):
    """Return (data, children) for a single-node tree with one leaf AABB."""
    data = np.array([list(box)], dtype=np.float64)
    children = np.array([[-1, -1, -1, -1]], dtype=np.int64)
    return data, children


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

INF = float("inf")


class LosKernelTest(unittest.TestCase):
    def test_single_node(self):
        data = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.float64)
        children = np.array([[-1, -1, -1, -1]], dtype=np.int64)
        max_depth = 1

        px = -0.5
        py = 0.5
        pz = 0.5

        idx = 1.0
        idy = float(np.inf)
        idz = float(np.inf)

        stack_size = 3 * max_depth + 4
        stack = np.empty(stack_size, dtype=np.int64)

        # Expect a hit.
        result = _los_kernel(
            data,
            children,
            px,
            py,
            pz,
            idx,
            idy,
            idz,
            10.0,
            0.0,
            stack,
        )
        self.assertFalse(math.isinf(result))

        # Expect no hit.
        result = _los_kernel(
            data,
            children,
            px,
            py,
            pz,
            -idx,
            idy,
            idz,
            10.0,
            0.0,
            stack,
        )
        self.assertTrue(math.isinf(result))

    def test_multiple_nodes(self):
        # fmt: off
        data = np.array(
            [
                [-2, -2, 0, 1, 1, 1.0],
                [ 0,  0, 0, 1, 1, 1.0],
                [-2, -2, 0, 0, 0, 0.5],
                [ 0, -2, 0, 1, 0, 1.0],
                [-2,  1, 0, 0, 0, 0.5],
            ]
        )
        children = np.array(
            [
                [ 1,  2,  3,  4],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ]
        )
        # fmt: on

        # Expect no hit.
        px = -1
        py = 1
        pz = 0.75

        idx = np.inf
        idy = -1
        idz = np.inf

        max_depth = 2
        stack_size = 3 * max_depth + 4
        stack = np.empty(stack_size, dtype=np.int64)

        result = _los_kernel(
            data,
            children,
            px,
            py,
            pz,
            -idx,
            idy,
            idz,
            10.0,
            0.0,
            stack,
        )
        self.assertTrue(math.isinf(result))

        # Expect hit.
        px = 0.5

        result = _los_kernel(
            data,
            children,
            px,
            py,
            pz,
            -idx,
            idy,
            idz,
            10.0,
            0.0,
            stack,
        )
        self.assertFalse(math.isinf(result))


class TestLosKernelSingleLeaf(unittest.TestCase):
    """Tests on a single-leaf tree: one AABB, no children."""

    # ------------------------------------------------------------------
    # Basic hit / miss along X axis
    # ------------------------------------------------------------------
    def test_ray_hits_leaf_from_left(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray travels in +X, starts left of box, aimed right → hit
        result = _los_kernel(
            data, children, -0.5, 0.5, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertFalse(math.isinf(result), "Ray should be blocked by the leaf AABB")

    def test_ray_misses_leaf_going_away(self):
        data, children = _single_leaf()
        stack = _make_stack(1)
        # Ray travels in -X, starts left of box, aimed further left → miss
        result = _los_kernel(
            data, children, -0.5, 0.5, 0.5, -1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertTrue(
            math.isinf(result), "Ray aimed away from box should have clear LOS"
        )

    def test_ray_misses_leaf_offset_in_y(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray travels in +X but y=2.0 is outside the [0,1] slab
        result = _los_kernel(
            data, children, -0.5, 2.0, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertTrue(math.isinf(result), "Ray offset above box in Y should miss")

    # ------------------------------------------------------------------
    # Ray origin inside the AABB
    # ------------------------------------------------------------------
    def test_ray_origin_inside_aabb_hits(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Origin is inside the box; any direction should still intersect
        result = _los_kernel(
            data, children, 0.5, 0.5, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertFalse(
            math.isinf(result), "Ray originating inside the leaf AABB should be blocked"
        )

    # ------------------------------------------------------------------
    # t_max cutoff
    # ------------------------------------------------------------------
    def test_t_max_cuts_ray_before_box(self):
        data, children = _single_leaf()  # box starts at x=0
        stack = _make_stack(1)
        # Ray origin at x=-5, travelling +X, box is 5 units away.
        # t_max=3 means the ray stops at x=-2, before the box.
        result = _los_kernel(
            data, children, -5.0, 0.5, 0.5, 1.0, INF, INF, 3.0, 0.0, stack
        )
        self.assertTrue(math.isinf(result), "Ray should not reach the box within t_max")

    def test_t_max_reaches_box(self):
        data, children = _single_leaf()
        stack = _make_stack(1)
        # Same setup but t_max=10 → ray reaches box
        result = _los_kernel(
            data, children, -5.0, 0.5, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertFalse(
            math.isinf(result), "Ray should reach and be blocked by the box"
        )

    # ------------------------------------------------------------------
    # Diagonal ray
    # ------------------------------------------------------------------
    def test_diagonal_ray_hits(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # 45-degree ray in XY plane aimed at box centre (0.5, 0.5)
        inv_sqrt2 = 1.0 / math.sqrt(0.5)
        result = _los_kernel(
            data, children, -1.0, -1.0, 0.5, inv_sqrt2, inv_sqrt2, INF, 10.0, 0.0, stack
        )
        self.assertFalse(
            math.isinf(result), "Diagonal ray aimed at box should be blocked"
        )

    def test_diagonal_ray_misses(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Same diagonal direction but offset so it passes above the box
        inv_sqrt2 = 1.0 / math.sqrt(0.5)
        result = _los_kernel(
            data, children, -1.0, 2.0, 0.5, inv_sqrt2, inv_sqrt2, INF, 10.0, 0.0, stack
        )
        self.assertTrue(math.isinf(result), "Diagonal ray above box should miss")

    # ------------------------------------------------------------------
    # Negative direction components
    # ------------------------------------------------------------------
    def test_negative_direction_hits(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray starts to the right of the box, travels in -X direction
        result = _los_kernel(
            data, children, 2.0, 0.5, 0.5, -1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertFalse(
            math.isinf(result), "Negative-direction ray through box should be blocked"
        )

    # ------------------------------------------------------------------
    # Axis-aligned ray parallel to a slab face but outside (IEEE-754 path)
    # ------------------------------------------------------------------
    def test_axis_aligned_ray_parallel_outside(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray moves purely in X (idy=idz=inf) but y=2.0 is outside [0,1]
        result = _los_kernel(
            data, children, -1.0, 2.0, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertTrue(
            math.isinf(result), "Axis-aligned ray outside a slab should produce a miss"
        )

    # ------------------------------------------------------------------
    # Grazing ray (hits exactly on a face)
    # ------------------------------------------------------------------
    def test_grazing_ray_on_face(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray travels in +X at y=0 (exactly on the y=0 face of the box)
        # Conservative behaviour: should still register as a hit
        result = _los_kernel(
            data, children, -1.0, 0.0, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertFalse(
            math.isinf(result),
            "Grazing ray on a face should count as a hit (conservative)",
        )

    # ------------------------------------------------------------------
    # t_min — skip self-intersections at the ray origin
    # ------------------------------------------------------------------
    def test_origin_inside_aabb_cleared_by_t_min(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Origin is inside the box, but t_min=0.5 skips the self-intersection
        # (t_enter=0 < t_min=0.5 → ignored).  No other hit exists → clear LOS.
        result = _los_kernel(
            data, children, 0.5, 0.5, 0.5, 1.0, INF, INF, 10.0, 0.5, stack
        )
        self.assertTrue(
            math.isinf(result), "t_min > 0 should skip the self-intersection at origin"
        )

    def test_origin_inside_aabb_still_blocked_at_t_min_zero(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # With t_min=0 the self-intersection at t_enter=0 is NOT skipped.
        result = _los_kernel(
            data, children, 0.5, 0.5, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertFalse(
            math.isinf(result), "t_min=0 should still report origin-inside as blocked"
        )

    def test_external_hit_not_skipped_by_small_t_min(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray from x=-5 hits the box at t_enter≈5.  t_min=0.5 is well below
        # that entry distance so the hit must still be reported.
        result = _los_kernel(
            data, children, -5.0, 0.5, 0.5, 1.0, INF, INF, 10.0, 0.5, stack
        )
        self.assertFalse(
            math.isinf(result), "Hit at t≈5 should not be skipped by t_min=0.5"
        )

    def test_returned_t_value_matches_entry_distance(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray from x=-5 travelling +X.  X slab entry: (0 - (-5)) / 1 = 5.0
        result = _los_kernel(
            data, children, -5.0, 0.5, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertAlmostEqual(
            result, 5.0, places=10, msg="Returned t should equal slab entry distance"
        )

    def test_has_line_of_sight_with_t_min_clears_origin_inside(self):
        data = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        children = np.array([[-1, -1, -1, -1]], dtype=np.int64)
        tree = HbvTree(data, children, Point(lat=0, lon=0, alt=0))
        ray = Ray(p_start=(0.5, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10.0)
        self.assertFalse(tree.has_line_of_sight(ray), "Default t_min=0 is blocked")
        self.assertTrue(
            tree.has_line_of_sight(ray, t_min=0.5), "t_min=0.5 skips self-intersection"
        )


class TestLosKernelTree(unittest.TestCase):
    """Tests on a small complete quadtree (depth-2, 1 root + 4 leaves)."""

    def _build_tree(self):
        # Root covers [-2, -2, 0] → [2, 2, 1]
        # Four children, each a leaf in one quadrant
        # fmt: off
        data = np.array(
            [
                [-2.0, -2.0, 0.0, 2.0, 2.0, 1.0],  # 0: root
                [-2.0, -2.0, 0.0, 0.0, 0.0, 1.0],  # 1: SW leaf
                [ 0.0, -2.0, 0.0, 2.0, 0.0, 1.0],  # 2: SE leaf
                [-2.0,  0.0, 0.0, 0.0, 2.0, 1.0],  # 3: NW leaf
                [ 0.0,  0.0, 0.0, 2.0, 2.0, 1.0],  # 4: NE leaf
            ],
            dtype=np.float64,
        )
        children = np.array(
            [
                [1, 2, 3, 4],    # root → four leaf children
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ],
            dtype=np.int64,
        )
        # fmt: on
        return data, children

    # ------------------------------------------------------------------
    # Root miss (ray doesn't touch root AABB at all)
    # ------------------------------------------------------------------
    def test_ray_misses_root_entirely(self):
        data, children = self._build_tree()
        stack = _make_stack(2)
        # Ray starts and stays at z=5 (outside [0,1] z-slab of every node)
        result = _los_kernel(
            data, children, -3.0, 0.0, 5.0, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertTrue(
            math.isinf(result),
            "Ray missing the root should return clear LOS immediately",
        )

    # ------------------------------------------------------------------
    # Ray hits root but misses all leaves
    # ------------------------------------------------------------------
    def test_ray_hits_root_misses_all_leaves(self):
        data, children = self._build_tree()
        stack = _make_stack(2)
        # Root spans [-2,2] in X and Y; leaves only cover |y|<2, |x|<2.
        # Send a ray through the root AABB that is entirely in the Z slab
        # but along a path where no individual leaf AABB is intersected.
        # We do this by making t_max very small so the ray exits the root
        # before reaching any leaf centre.
        # Actually: root IS the only node whose AABB the ray intersects, but
        # because it has children the kernel only returns False on leaf hits.
        # So if the root AABB is hit and we traverse to children, but the
        # ray's Y stays outside all leaf Y-slabs, all children miss.
        # Children Y slabs: [−2,0] and [0,2]. A ray at Y=−3 misses both.
        result = _los_kernel(
            data, children, -3.0, -3.0, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertTrue(
            math.isinf(result),
            "Ray inside root but outside all leaf AABBs should be clear",
        )

    # ------------------------------------------------------------------
    # Selective leaf hit
    # ------------------------------------------------------------------
    def test_ray_hits_only_sw_leaf(self):
        data, children = self._build_tree()
        stack = _make_stack(2)
        # Ray aimed at SW leaf centre (-1, -1)
        result = _los_kernel(
            data, children, -3.0, -1.0, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertFalse(math.isinf(result), "Ray through SW leaf should be blocked")

    def test_ray_hits_only_ne_leaf(self):
        data, children = self._build_tree()
        stack = _make_stack(2)
        # Ray aimed at NE leaf centre (1, 1)
        result = _los_kernel(
            data, children, -3.0, 1.0, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertFalse(math.isinf(result), "Ray through NE leaf should be blocked")

    # ------------------------------------------------------------------
    # t_max stops ray inside root but before leaves
    # ------------------------------------------------------------------
    def test_t_max_stops_before_leaf(self):
        data, children = self._build_tree()
        stack = _make_stack(2)
        # Root starts at x=-2. Ray origin at x=-10.
        # The root AABB x-slab starts at t=8, so t_max=7 won't even reach it.
        result = _los_kernel(
            data, children, -10.0, 0.0, 0.5, 1.0, INF, INF, 7.0, 0.0, stack
        )
        self.assertTrue(
            math.isinf(result), "Ray stopped by t_max before root should have clear LOS"
        )

    # ------------------------------------------------------------------
    # Boundary between two leaves
    # ------------------------------------------------------------------
    def test_ray_on_leaf_boundary(self):
        data, children = self._build_tree()
        # Ray travels along y=0, the boundary between SW/SE and NW/NE leaves.
        # Conservative: the ray grazes both SW and SE leaf faces; either or
        # both may register as a hit (implementation-defined), so we just
        # verify the call completes without error and returns a float.
        stack2 = _make_stack(2)
        result = _los_kernel(
            data, children, -3.0, 0.0, 0.5, 1.0, INF, INF, 10.0, 0.0, stack2
        )
        self.assertIsInstance(result, float)
        self.assertFalse(math.isinf(result))


class TestLosKernelNaNBehaviour(unittest.TestCase):
    """
    The docstring explicitly acknowledges NaN (ray origin exactly on a slab
    face) is not guarded against, but guarantees no false negatives — at
    worst one extra subtree is visited.  We test that the function at least
    terminates and returns a float.
    """

    def test_nan_inv_direction_terminates(self):
        data, children = _single_leaf = (
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float64),
            np.array([[-1, -1, -1, -1]], dtype=np.int64),
        )
        stack = _make_stack(1)
        # Ray origin on the x=0 face of the box → NaN in x slab calculation
        result = _los_kernel(
            data, children, 0.0, 0.5, 0.5, 1.0, INF, INF, 10.0, 0.0, stack
        )
        self.assertIsInstance(result, float, "NaN path must still return a float")
        self.assertFalse(math.isinf(result))


def _make_grid(
    h: int,
    w: int,
    *,
    x_min=0.0,
    y_min=0.0,
    z_min=0.0,
    x_max=1.0,
    y_max=1.0,
    z_max=1.0,
) -> np.ndarray:
    """Return a (H, W, 6) array where every cell has the same AABB."""
    cell = np.array([x_min, y_min, z_min, x_max, y_max, z_max], dtype=np.float64)
    return np.broadcast_to(cell, (h, w, 6)).copy()


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class TestBuildHigherLevelShape(unittest.TestCase):
    def test_grid_shape(self):
        # Even grid shape.
        out = build_higher_level(_make_grid(4, 4))
        self.assertEqual(out.shape, (2, 2, 6))

        # Even non-square.
        out = build_higher_level(_make_grid(4, 8))
        self.assertEqual(out.shape, (2, 4, 6))

        # Odd height drops last row.
        # 5×4 → only the first 4 rows are used → 2×2 output
        out = build_higher_level(_make_grid(5, 4))
        self.assertEqual(out.shape, (2, 2, 6))

        # 5×2 input: the 5th row is dropped. Output must equal that of a 4×2 input
        # with the same first four rows, regardless of what the 5th row contains.
        base = np.random.default_rng(0).random((4, 2, 6))
        with_extra = np.concatenate(
            [base, np.full((1, 2, 6), fill_value=999.0)], axis=0
        )
        self.assertTrue(
            np.allclose(
                build_higher_level(base),
                build_higher_level(with_extra),
            )
        )

        # Odd width drops last column.
        out = build_higher_level(_make_grid(4, 5))
        self.assertEqual(out.shape, (2, 2, 6))

        base = np.random.default_rng(1).random((2, 4, 6))
        with_extra = np.concatenate(
            [base, np.full((2, 1, 6), fill_value=999.0)], axis=1
        )
        self.assertTrue(
            np.allclose(
                build_higher_level(base),
                build_higher_level(with_extra),
            )
        )

        # Odd height and width drop last row and column.
        out = build_higher_level(_make_grid(5, 5))
        self.assertEqual(out.shape, (2, 2, 6))

        # Minimum 2x2 case.
        out = build_higher_level(_make_grid(2, 2))
        self.assertEqual(out.shape, (1, 1, 6))

        # 1x1 case returns empty array.
        # 1//2 == 0 in each spatial dimension
        out = build_higher_level(_make_grid(1, 1))
        self.assertEqual(out.shape, (0, 0, 6))

    def test_correct_pooling(self):
        """
        Each output cell must be the tightest AABB enclosing its 4 input children.
        The 6 channels are: [x_min, y_min, z_min, x_max, y_max, z_max].
        min-pool is applied to channels 0-2 (lower bounds).
        max-pool is applied to channels 3-5 (upper bounds).
        """

        def _four_distinct_cells():
            """
            2×2 input where each cell has a distinct, non-overlapping AABB.
            Arranged so the correct parent AABB is unambiguous.

            cell (0,0): x∈[0,1], y∈[0,1], z∈[0,1]
            cell (0,1): x∈[1,2], y∈[0,1], z∈[0,1]
            cell (1,0): x∈[0,1], y∈[1,2], z∈[0,1]
            cell (1,1): x∈[1,2], y∈[1,2], z∈[0,1]

            Expected parent: x∈[0,2], y∈[0,2], z∈[0,1]
            """
            grid = np.array(
                [
                    [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 2.0, 1.0, 1.0]],
                    [[0.0, 1.0, 0.0, 1.0, 2.0, 1.0], [1.0, 1.0, 0.0, 2.0, 2.0, 1.0]],
                ],
                dtype=np.float64,
            )
            return grid

        out = build_higher_level(_four_distinct_cells())

        # Lower bounds use min.
        # x_min, y_min, z_min should all be 0.0 (minimum across the 4 cells)
        self.assertTrue(np.allclose(out[0, 0, :3], [0.0, 0.0, 0.0]))

        # Upper bounds use max.
        # x_max, y_max, z_max should all be 2.0 (maximum across the 4 cells)
        self.assertTrue(np.allclose(out[0, 0, 3:], [2.0, 2.0, 1.0]))

        # If axes were swapped, lower bounds would be too large and upper
        # bounds too small — the AABB would fail to enclose its children.
        self.assertTrue(
            np.all(out[..., :3] <= out[..., 3:]),
            "Every lower bound must be <= the corresponding upper bound",
        )

    def test_axes_are_not_mixed_up(self):
        def _4x4_quadrant_grid():
            """
            4×4 grid where each 2×2 quadrant has a distinct z range so any
            cross-quadrant bleed is immediately visible.

            NW patch (rows 0-1, cols 0-1): z∈[0, 1]
            NE patch (rows 0-1, cols 2-3): z∈[2, 3]
            SW patch (rows 2-3, cols 0-1): z∈[4, 5]
            SE patch (rows 2-3, cols 2-3): z∈[6, 7]

            All cells share x∈[0,1], y∈[0,1] so only z distinguishes patches.
            """
            grid = np.zeros((4, 4, 6), dtype=np.float64)
            grid[:, :, 3] = 1.0  # x_max
            grid[:, :, 4] = 1.0  # y_max
            for (rs, re, cs, ce), (z_lo, z_hi) in [
                ((0, 2, 0, 2), (0.0, 1.0)),
                ((0, 2, 2, 4), (2.0, 3.0)),
                ((2, 4, 0, 2), (4.0, 5.0)),
                ((2, 4, 2, 4), (6.0, 7.0)),
            ]:
                grid[rs:re, cs:ce, 2] = z_lo
                grid[rs:re, cs:ce, 5] = z_hi
            return grid

        # Use asymmetric extents so a channel mix-up (e.g. y_min pooled into
        # x_min) would produce a wrong value.
        grid = np.array(
            [
                [
                    [1.0, 10.0, 100.0, 2.0, 20.0, 200.0],
                    [3.0, 30.0, 300.0, 4.0, 40.0, 400.0],
                ],
                [
                    [5.0, 50.0, 500.0, 6.0, 60.0, 600.0],
                    [7.0, 70.0, 700.0, 8.0, 80.0, 800.0],
                ],
            ],
            dtype=np.float64,
        )
        out = build_higher_level(grid)
        expected = np.array([1.0, 10.0, 100.0, 8.0, 80.0, 800.0])
        self.assertTrue(np.allclose(out[0, 0], expected))

        # Correct spatial grouping of cells.
        # pool2d groups cells into non-overlapping 2×2 patches.
        # A 4×4 input produces 4 independent output cells; each must reflect
        # only its own 2×2 patch and not bleed into neighbours.
        out = build_higher_level(_4x4_quadrant_grid())
        self.assertEqual(out.shape, (2, 2, 6))
        self.assertTrue(np.allclose(out[0, 0, 2:6:3], [0.0, 1.0]))  # NW: z∈[0,1]
        self.assertTrue(np.allclose(out[0, 1, 2:6:3], [2.0, 3.0]))  # NE: z∈[2,3]
        self.assertTrue(np.allclose(out[1, 0, 2:6:3], [4.0, 5.0]))  # SW: z∈[4,5]
        self.assertTrue(np.allclose(out[1, 1, 2:6:3], [6.0, 7.0]))  # SE: z∈[6,7]

        # No cross-patch-bleed.
        out = build_higher_level(_4x4_quadrant_grid())
        # If any patch bled into a neighbour the z ranges would overlap
        z_mins = out[:, :, 2]
        z_maxs = out[:, :, 5]
        self.assertLess(z_maxs[0, 0], z_mins[0, 1], "NW z_max must be < NE z_min")
        self.assertLess(z_maxs[0, 0], z_mins[1, 0], "NW z_max must be < SW z_min")

    def test_edge_and_special_cases(self):
        # Uniform grid: AABB out = AABB in
        grid = _make_grid(
            4,
            4,
            x_min=1.0,
            y_min=2.0,
            z_min=3.0,
            x_max=4.0,
            y_max=5.0,
            z_max=6.0,
        )
        out = build_higher_level(grid)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertTrue(np.allclose(out[0, 0], expected))
        self.assertTrue(np.allclose(out[1, 1], expected))

        # Negative coordinates.
        grid = _make_grid(
            2,
            2,
            x_min=-5.0,
            y_min=-3.0,
            z_min=-1.0,
            x_max=-4.0,
            y_max=-2.0,
            z_max=0.0,
        )
        out = build_higher_level(grid)
        self.assertTrue(
            np.allclose(out[0, 0], [-5.0, -3.0, -1.0, -4.0, -2.0, 0.0]),
        )

        # AABB of volume zero should not raise an error.
        grid = _make_grid(
            2, 2, x_min=1.0, y_min=1.0, z_min=1.0, x_max=1.0, y_max=1.0, z_max=1.0
        )
        out = build_higher_level(grid)
        self.assertTrue(np.allclose(out[0, 0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        # Test output type.
        grid = _make_grid(2, 2)
        out = build_higher_level(grid)
        self.assertEqual(out.dtype, np.float64)

        # Test output is c-contiguous.
        grid = _make_grid(4, 4)
        out = build_higher_level(grid)
        self.assertTrue(out.flags["C_CONTIGUOUS"])


class TestBuildBoundingBoxes(unittest.TestCase):
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_grid(nlats=3, nlons=3, spacing=1.0, lat0=0.0, lon0=0.0):
        """Return (lats, lons, flat-zero data) for a regular grid."""
        lats = np.array([lat0 + i * spacing for i in range(nlats)])
        lons = np.array([lon0 + j * spacing for j in range(nlons)])
        data = np.zeros((nlats, nlons))
        return lats, lons, data

    @staticmethod
    def _identity_geodetic_to_cartesian(lat, lon, alt):
        """
        Trivial stub: ECEF ≈ (lon, lat, alt).
        Simple enough to compute expected min/max by hand.
        """
        return np.array([lon, lat, alt], dtype=float)

    @staticmethod
    def _fake_ecef_to_enu_multiple(arr):
        """
        Trivial stub: ENU = (z, x, y), where (x, y, z) are ECEF coordinates.
        Simple enough to compute expected min/max by hand.
        """
        return np.array(arr)[:, [1, 0, 2]]

    # ------------------------------------------------------------------
    # Output shape / dtype
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Result must be (Nlats, Nlons, 6)."""
        lats, lons, data = self._make_grid(nlats=4, nlons=5)
        with patch(
            "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
            side_effect=self._identity_geodetic_to_cartesian,
        ):
            result, _ = build_bounding_boxes(lats, lons, data)
        self.assertEqual(result.shape, (4, 5, 6))

    def test_output_dtype_is_float(self):
        """Output array should contain floating-point values."""
        lats, lons, data = self._make_grid()
        with patch(
            "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
            side_effect=self._identity_geodetic_to_cartesian,
        ):
            result, _ = build_bounding_boxes(lats, lons, data)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    # ------------------------------------------------------------------
    # Bounding-box geometry invariants
    # ------------------------------------------------------------------

    def test_min_le_max_for_all_cells(self):
        """x_min ≤ x_max, y_min ≤ y_max, z_min ≤ z_max for every cell."""
        lats, lons, data = self._make_grid(nlats=3, nlons=3)
        with patch(
            "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
            side_effect=self._identity_geodetic_to_cartesian,
        ):
            result, _ = build_bounding_boxes(lats, lons, data)
        self.assertTrue(np.all(result[..., :3] <= result[..., 3:]))

    def test_bbox_values_match_manual_calculation(self):
        """
        Verify all four AABB cells for a 2×2 grid with uniform elevation=100.

        With the identity stub (lat, lon, alt) → [lon, lat, alt] and
        half_spacing=0.5, alt_below = clip(100,0,None) - 30 = 70:

        Cell [0,0]: lat=0, lon=0 → x∈{-0.5,0.5}, y∈{-0.5,0.5}, z∈{70,100}
        Cell [0,1]: lat=0, lon=1 → x∈{ 0.5,1.5}, y∈{-0.5,0.5}, z∈{70,100}
        Cell [1,0]: lat=1, lon=0 → x∈{-0.5,0.5}, y∈{ 0.5,1.5}, z∈{70,100}
        Cell [1,1]: lat=1, lon=1 → x∈{ 0.5,1.5}, y∈{ 0.5,1.5}, z∈{70,100}

        Note: x tracks longitude and y tracks latitude because the stub
        returns [lon, lat, alt] rather than a true ECEF vector.
        """
        ALT = 100.0
        ALT_BELOW = ALT - 30  # mirrors clip(alt, 0, None) - 30

        lats = np.array([0.0, 1.0])
        lons = np.array([0.0, 1.0])
        data = np.full((2, 2), ALT)

        with patch(
            "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
            side_effect=self._identity_geodetic_to_cartesian,
        ):
            mock_ecef_to_enu = MagicMock()
            mock_ecef_to_enu.ecef_to_enu_multiple.side_effect = (
                self._fake_ecef_to_enu_multiple
            )
            with patch(
                "theia.coordinates.EcefToEnuTransformer", return_value=mock_ecef_to_enu
            ):
                result, _ = build_bounding_boxes(lats, lons, data, n_jobs=1)

        expected = np.array(
            [
                [
                    [-0.5, -0.5, ALT_BELOW, 0.5, 0.5, ALT],
                    [-0.5, 0.5, ALT_BELOW, 0.5, 1.5, ALT],
                ],
                [
                    [0.5, -0.5, ALT_BELOW, 1.5, 0.5, ALT],
                    [0.5, 0.5, ALT_BELOW, 1.5, 1.5, ALT],
                ],
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_alt_below_positive_elevation(self):
        """
        For positive alt, alt_below = alt - 30.
        With the identity stub z_min should equal alt - 30.
        """
        lats = np.array([0.0, 1.0])
        lons = np.array([0.0, 1.0])
        alt = 500.0
        data = np.full((2, 2), alt)

        with patch(
            "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
            side_effect=self._identity_geodetic_to_cartesian,
        ):
            mock_ecef_to_enu = MagicMock()
            mock_ecef_to_enu.ecef_to_enu_multiple.side_effect = (
                self._fake_ecef_to_enu_multiple
            )
            with patch(
                "theia.coordinates.EcefToEnuTransformer", return_value=mock_ecef_to_enu
            ):
                result, _ = build_bounding_boxes(lats, lons, data, n_jobs=1)

        z_min = result[:, :, 2]
        np.testing.assert_allclose(z_min, alt - 30.0)

    def test_alt_below_zero_elevation(self):
        """
        For alt=0 (sea level), clip(0, 0, None)=0, so alt_below = -30.
        z_min for the identity stub should be -30.
        """
        lats = np.array([0.0, 1.0])
        lons = np.array([0.0, 1.0])
        alt = 0
        data = np.full((2, 2), alt)

        with patch(
            "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
            side_effect=self._identity_geodetic_to_cartesian,
        ):
            mock_ecef_to_enu = MagicMock()
            mock_ecef_to_enu.ecef_to_enu_multiple.side_effect = (
                self._fake_ecef_to_enu_multiple
            )
            with patch(
                "theia.coordinates.EcefToEnuTransformer", return_value=mock_ecef_to_enu
            ):
                result, _ = build_bounding_boxes(lats, lons, data, n_jobs=1)

        z_min = result[:, :, 2]
        self.assertTrue(np.allclose(z_min, alt - 30.0))

    # ------------------------------------------------------------------
    # Spacing / assertion guard
    # ------------------------------------------------------------------

    def test_unequal_lat_lon_spacing_raises(self):
        """Unequal lat/lon spacing must trigger the internal assertion."""
        lats = np.array([0.0, 1.0, 2.0])
        lons = np.array([0.0, 2.0, 4.0])  # spacing=2 ≠ spacing=1
        data = np.zeros((3, 3))

        with self.assertRaises(AssertionError):
            build_bounding_boxes(lats, lons, data)

    def test_equal_spacing_does_not_raise(self):
        """Equal lat/lon spacing must not raise."""
        lats, lons, data = self._make_grid(spacing=0.5)
        try:
            with patch(
                "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
                side_effect=self._identity_geodetic_to_cartesian,
            ):
                build_bounding_boxes(lats, lons, data)
        except AssertionError:
            self.fail("build_bounding_boxes raised AssertionError for equal spacing")

    # ------------------------------------------------------------------
    # Call count – geodetic_to_cartesian is called 8× per cell
    # ------------------------------------------------------------------

    def test_geodetic_to_cartesian_call_count(self):
        """The transform must be called exactly 8 times per grid cell."""
        nlats, nlons = 2, 3
        lats, lons, data = self._make_grid(nlats=nlats, nlons=nlons)

        with patch(
            "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
            side_effect=self._identity_geodetic_to_cartesian,
        ) as mock_fn:
            build_bounding_boxes(lats, lons, data, n_jobs=1)

        # + 1: Transforming the origin of the ENU system
        expected_calls = 8 * nlats * nlons + 1
        self.assertEqual(mock_fn.call_count, expected_calls)

    # ------------------------------------------------------------------
    # Varying elevation across cells
    # ------------------------------------------------------------------

    def test_different_elevations_per_cell(self):
        """Each cell's z_max should equal that cell's elevation."""
        lats = np.array([0.0, 1.0, 2.0])
        lons = np.array([0.0, 1.0, 2.0])
        data = np.array(
            [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0], [700.0, 800.0, 900.0]]
        )

        with patch(
            "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
            side_effect=self._identity_geodetic_to_cartesian,
        ):
            mock_ecef_to_enu = MagicMock()
            mock_ecef_to_enu.ecef_to_enu_multiple.side_effect = (
                self._fake_ecef_to_enu_multiple
            )
            with patch(
                "theia.coordinates.EcefToEnuTransformer", return_value=mock_ecef_to_enu
            ):
                result, _ = build_bounding_boxes(lats, lons, data, n_jobs=1)

        # z_max (index 5) equals the surface elevation for each cell
        np.testing.assert_allclose(result[..., 5], data)

    # ------------------------------------------------------------------
    # Correct corners are passed to the transform
    # ------------------------------------------------------------------

    def test_correct_corner_coordinates_passed(self):
        """
        For cell (lat=1, lon=1) with half_spacing=0.5, verify that the
        8 exact (lat±0.5, lon±0.5, alt / alt_below) corner combos are
        all present in the mock call arguments.
        """
        lats = np.array([1.0, 2.0])
        lons = np.array([1.0, 2.0])
        alt = 50.0
        data = np.full((2, 2), alt)

        with patch(
            "theia.coordinates.CoordinateTransformations.geodetic_to_cartesian",
            side_effect=self._identity_geodetic_to_cartesian,
        ) as mock_fn:
            mock_ecef_to_enu = MagicMock()
            mock_ecef_to_enu.ecef_to_enu_multiple.side_effect = (
                self._fake_ecef_to_enu_multiple
            )
            with patch(
                "theia.coordinates.EcefToEnuTransformer", return_value=mock_ecef_to_enu
            ):
                build_bounding_boxes(lats, lons, data, n_jobs=1)

        # Extract calls for cell (0,0): first 8 calls
        first_8 = mock_fn.call_args_list[:8]
        actual = {(c.args[0], c.args[1], c.args[2]) for c in first_8}

        hs = 0.5
        lat, lon = 1.0, 1.0
        alt_below = alt - 30.0
        expected = {
            (lat - hs, lon - hs, alt),
            (lat + hs, lon - hs, alt),
            (lat - hs, lon + hs, alt),
            (lat + hs, lon + hs, alt),
            (lat - hs, lon - hs, alt_below),
            (lat + hs, lon - hs, alt_below),
            (lat - hs, lon + hs, alt_below),
            (lat + hs, lon + hs, alt_below),
        }
        self.assertEqual(actual, expected)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_hgt_tile(lat: int, lon: int, fill_value: float = 0.0) -> np.ndarray:
    """Return a synthetic 1201×1201 HGT tile (standard SRTM-3 resolution)."""
    return np.full((1201, 1201), fill_value, dtype=np.float32)


def make_hgt_tile_gradient(lat: int, lon: int) -> np.ndarray:
    """Return a tile whose values encode (lat, lon) for traceability."""
    tile = np.zeros((1201, 1201), dtype=np.float32)
    tile[:] = lat * 1000 + lon  # every cell = lat*1000 + lon
    return tile


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadSrtmBboxes(unittest.TestCase):
    """Unit tests for load_srtm_bboxes.

    load_hgt_file is mocked throughout so no real .hgt files are needed.
    """

    # ------------------------------------------------------------------
    # Return-type / shape invariants
    # ------------------------------------------------------------------

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_returns_three_arrays(self, _mock):
        result = load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_output_shapes_are_square_power_of_two(self, _mock):
        lats, lons, data = load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        N = len(lats)
        self.assertEqual(len(lons), N)
        self.assertEqual(data.shape, (N, N))
        # N must be a power of two
        self.assertTrue(
            np.log2(N).is_integer(), f"Expected power-of-two side length, got {N}"
        )

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_lats_and_lons_are_1d(self, _mock):
        lats, lons, data = load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        self.assertEqual(lats.ndim, 1)
        self.assertEqual(lons.ndim, 1)

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_data_is_2d(self, _mock):
        lats, lons, data = load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        self.assertEqual(data.ndim, 2)

    # ------------------------------------------------------------------
    # Coordinate range / ordering
    # ------------------------------------------------------------------

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_lats_start_at_floored_lat_start(self, _mock):
        lats, lons, _ = load_srtm_bboxes(47.3, 48.7, 11.0, 12.0)
        self.assertAlmostEqual(lats[0], 47.0)  # floor(47.3)

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_lons_start_at_floored_lon_start(self, _mock):
        lats, lons, _ = load_srtm_bboxes(47.0, 48.0, 11.2, 12.8)
        self.assertAlmostEqual(lons[0], 11.0)  # floor(11.2)

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_lats_are_monotonically_increasing(self, _mock):
        lats, _, _ = load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        self.assertTrue(np.all(np.diff(lats) > 0))

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_lons_are_monotonically_increasing(self, _mock):
        _, lons, _ = load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        self.assertTrue(np.all(np.diff(lons) > 0))

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_lats_are_uniformly_spaced(self, _mock):
        lats, _, _ = load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        diffs = np.diff(lats)
        self.assertTrue(np.allclose(diffs, diffs[0]), "Latitude spacing is not uniform")

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_lons_are_uniformly_spaced(self, _mock):
        _, lons, _ = load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        diffs = np.diff(lons)
        self.assertTrue(
            np.allclose(diffs, diffs[0]), "Longitude spacing is not uniform"
        )

    # ------------------------------------------------------------------
    # Bounding-box snapping (non-integer inputs)
    # ------------------------------------------------------------------

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_fractional_bbox_is_snapped_to_integer_degrees(self, _mock):
        """Non-integer lat/lon bounds must be snapped before tiling."""

        # Fractional inputs — should snap to 47, 49, 11, 13
        lats, lons, _ = load_srtm_bboxes(47.4, 48.9, 11.1, 12.6)
        self.assertAlmostEqual(lats[0], 47.0)
        self.assertAlmostEqual(lons[0], 11.0)

    # ------------------------------------------------------------------
    # Multi-tile stitching: load_hgt_file call count
    # ------------------------------------------------------------------

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_single_tile_calls_load_hgt_once(self, mock_load):
        load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        self.assertEqual(mock_load.call_count, 1)

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_two_lon_tiles_calls_load_hgt_twice(self, mock_load):
        load_srtm_bboxes(47.0, 48.0, 11.0, 13.0)  # 1 lat × 2 lon tiles
        self.assertEqual(mock_load.call_count, 2)

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_two_lat_tiles_calls_load_hgt_twice(self, mock_load):
        load_srtm_bboxes(47.0, 49.0, 11.0, 12.0)  # 2 lat × 1 lon tiles
        self.assertEqual(mock_load.call_count, 2)

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_2x2_grid_calls_load_hgt_four_times(self, mock_load):
        load_srtm_bboxes(47.0, 49.0, 11.0, 13.0)  # 2 lat × 2 lon tiles
        self.assertEqual(mock_load.call_count, 4)

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_correct_tile_coordinates_requested(self, mock_load):
        """Verify that the correct (lat, lon) pairs are fetched."""

        load_srtm_bboxes(47.0, 49.0, 11.0, 13.0)
        called_args = {call.args for call in mock_load.call_args_list}
        expected = {(47, 11), (47, 12), (48, 11), (48, 12)}
        self.assertEqual(called_args, expected)

    # ------------------------------------------------------------------
    # Padding: zeros fill the padded region
    # ------------------------------------------------------------------

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile_gradient)
    def test_padded_region_is_zero(self, _mock):
        """Cells outside the real data area must be zero-padded."""

        lats, lons, data = load_srtm_bboxes(47.0, 48.0, 11.0, 13.0)
        # For a 1-lat × 2-lon input the real columns are narrower than N
        # when N is the next power-of-two; excess columns should be zero.
        # The real data should be in the slice data[:1201, :2*1201] because
        # the test data has side length 1201 per tile and consecutive tiles share
        # the first row or column, respectively.
        real_data = data[:1201, : 2 * 1201 - 1]
        self.assertTrue(np.all(real_data != 0), "Real data should not be zero")
        self.assertTrue(
            np.all(data[1201:, :] == 0),
            "Padded rows should be zero",
        )
        self.assertTrue(
            np.all(data[:, 2 * 1201 - 1 :] == 0),
            "Padded columns should be zero",
        )

    # ------------------------------------------------------------------
    # Edge case: zero-fill for flat terrain
    # ------------------------------------------------------------------

    @patch(
        "theia.terrain.load_hgt_file",
        side_effect=lambda lat, lon: make_hgt_tile(lat, lon, 0.0),
    )
    def test_flat_terrain_returns_zero_data(self, _mock):
        _, _, data = load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        self.assertTrue(np.all(data == 0.0))

    # ------------------------------------------------------------------
    # Internal assertions in the function must not fire
    # ------------------------------------------------------------------

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_no_assertion_error_single_tile(self, _mock):
        try:
            load_srtm_bboxes(47.0, 48.0, 11.0, 12.0)
        except AssertionError as exc:
            self.fail(f"load_srtm_bboxes raised AssertionError: {exc}")

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile)
    def test_no_assertion_error_multi_tile(self, _mock):
        try:
            load_srtm_bboxes(47.0, 49.0, 11.0, 13.0)
        except AssertionError as exc:
            self.fail(f"load_srtm_bboxes raised AssertionError: {exc}")

    # ------------------------------------------------------------------
    # Multi-tile ordering and deduplication
    # ------------------------------------------------------------------

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile_gradient)
    def test_multi_lat_tile_ordering(self, _mock):
        """
        Southernmost tile data (lat=47, value≈47011) must appear at the lowest
        row indices; northernmost tile data (lat=48, value≈48011) at the last
        real-data row (index 2400 = 2*1201-1).  data[-1] is in the zero-padded
        region and must NOT be used for this comparison.
        """
        _, _, data = load_srtm_bboxes(47.0, 49.0, 11.0, 12.0)
        # 2 tiles × 1201 rows, 1 shared boundary → 2401 real rows
        last_real_row = 2 * 1201 - 2  # = 2400
        self.assertLess(
            data[0, 0],
            data[last_real_row, 0],
            "Row 0 should contain the southernmost (lower-valued) tile data",
        )

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile_gradient)
    def test_multi_lat_tile_no_duplicate_row(self, _mock):
        """
        Two adjacent 1201-row tiles share one boundary row.
        After stitching: 2*1201-1 = 2401 real rows.
        With the gradient mock (value = lat*1000+lon, never 0), the last real
        row (index 2400) is non-zero while row 2401 is zero-padded.
        """
        _, _, data = load_srtm_bboxes(47.0, 49.0, 11.0, 12.0)
        last_real_row = 2 * 1201 - 2  # = 2400
        self.assertGreater(
            data[last_real_row, 0],
            0,
            "Row 2400 should contain real (non-padded) tile data",
        )
        self.assertEqual(
            data[last_real_row + 1, 0],
            0.0,
            "Row 2401 should be zero-padded",
        )

    @patch("theia.terrain.load_hgt_file", side_effect=make_hgt_tile_gradient)
    def test_multi_lon_tile_ordering(self, _mock):
        """
        Westernmost tile data (lon=11, value≈47011) must appear at the lowest
        column indices; easternmost tile data (lon=12, value≈47012) at the last
        real-data column (index 2400 = 2*1201-1).  data[:, -1] is in the
        zero-padded region and must NOT be used for this comparison.
        """
        _, _, data = load_srtm_bboxes(47.0, 48.0, 11.0, 13.0)
        # 2 tiles × 1201 cols, 1 shared boundary → 2401 real cols
        last_real_col = 2 * 1201 - 2  # = 2400
        self.assertLess(
            data[0, 0],
            data[0, last_real_col],
            "Column 0 should contain the westernmost (lower-valued) tile data",
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _unit_leaf_grid(n: int) -> np.ndarray:
    """Return an (n, n, 6) grid of unit cubes with AABB [j, i, 0, j+1, i+1, 1]."""
    leaves = np.zeros((n, n, 6))
    for i in range(n):
        for j in range(n):
            leaves[i, j] = [j, i, 0.0, j + 1.0, i + 1.0, 1.0]
    return leaves


def _simple_2x2_tree() -> HbvTree:
    """
    2×2 grid of unit cubes tiling [0,2]×[0,2]×[0,1].

    Leaf layout (x increases right, y increases up):
        (0,1)=[0,1,0,1,2,1]  (1,1)=[1,1,0,2,2,1]
        (0,0)=[0,0,0,1,1,1]  (1,0)=[1,0,0,2,1,1]
    """
    return HbvTree.from_leaf_bboxes(_unit_leaf_grid(2), Point(lat=0, lon=0, alt=0))


# ===========================================================================
# 1. Helper function tests
# ===========================================================================


class TestDepthToNumNodes(unittest.TestCase):
    def test_depth_0(self):
        self.assertEqual(_depth_to_num_nodes(0), 0)

    def test_depth_1(self):
        # Root only
        self.assertEqual(_depth_to_num_nodes(1), 1)

    def test_depth_2(self):
        # 1 root + 4 children
        self.assertEqual(_depth_to_num_nodes(2), 5)

    def test_depth_3(self):
        # 1 + 4 + 16
        self.assertEqual(_depth_to_num_nodes(3), 21)

    def test_depth_4(self):
        # 1 + 4 + 16 + 64
        self.assertEqual(_depth_to_num_nodes(4), 85)

    def test_geometric_formula(self):
        for d in range(1, 8):
            expected = (4**d - 1) // 3
            with self.subTest(depth=d):
                self.assertEqual(_depth_to_num_nodes(d), expected)


class TestNumNodesToDepth(unittest.TestCase):
    def test_depth_1(self):
        self.assertEqual(_num_nodes_to_depth(1), 1)

    def test_depth_2(self):
        self.assertEqual(_num_nodes_to_depth(5), 2)

    def test_depth_3(self):
        self.assertEqual(_num_nodes_to_depth(21), 3)

    def test_roundtrip(self):
        for d in range(1, 8):
            with self.subTest(depth=d):
                self.assertEqual(_num_nodes_to_depth(_depth_to_num_nodes(d)), d)


class TestRay(unittest.TestCase):
    def test_basic_construction(self):
        r = Ray([0, 0, 0], [1, 0, 0], t_max=10.0)
        np.testing.assert_array_equal(r.p_start, [0, 0, 0])
        np.testing.assert_array_equal(r.direction, [1, 0, 0])
        self.assertEqual(r.t_max, 10.0)

    def test_zero_direction_raises(self):
        with self.assertRaises(ValueError):
            Ray([0, 0, 0], [0, 0, 0], t_max=1.0)

    def test_list_input_accepted(self):
        r = Ray([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 100.0)
        self.assertEqual(r.t_max, 100.0)


# ===========================================================================
# 3 & 4. Tree construction and structural tests
# ===========================================================================


class TestFromLeafBboxes(unittest.TestCase):
    def test_non_power_of_two_raises(self):
        with self.assertRaises(ValueError):
            HbvTree.from_leaf_bboxes(np.zeros((3, 3, 6)), Point(lat=0, lon=0, alt=0))

    def test_zero_size_raises(self):
        with self.assertRaises(ValueError):
            HbvTree.from_leaf_bboxes(np.zeros((0, 0, 6)), Point(lat=0, lon=0, alt=0))

    def test_2x2_node_count(self):
        # 2×2 leaves → depth 2 → 5 nodes
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(2), Point(lat=0, lon=0, alt=0))
        self.assertEqual(tree._data.shape, (5, 6))
        self.assertEqual(tree._children.shape, (5, 4))

    def test_4x4_node_count(self):
        # 4×4 leaves → depth 3 → 21 nodes
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(4), Point(lat=0, lon=0, alt=0))
        self.assertEqual(tree._data.shape[0], 21)

    def test_8x8_node_count(self):
        # 8×8 leaves → depth 4 → 85 nodes
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(8), Point(lat=0, lon=0, alt=0))
        self.assertEqual(tree._data.shape[0], 85)

    def test_root_aabb_2x2(self):
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(2), Point(lat=0, lon=0, alt=0))
        root = tree._data[0]
        # Leaves span x∈[0,2], y∈[0,2], z∈[0,1]
        self.assertAlmostEqual(root[0], 0.0)  # xmin
        self.assertAlmostEqual(root[1], 0.0)  # ymin
        self.assertAlmostEqual(root[2], 0.0)  # zmin
        self.assertAlmostEqual(root[3], 2.0)  # xmax
        self.assertAlmostEqual(root[4], 2.0)  # ymax
        self.assertAlmostEqual(root[5], 1.0)  # zmax

    def test_root_aabb_4x4(self):
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(4), Point(lat=0, lon=0, alt=0))
        root = tree._data[0]
        self.assertAlmostEqual(root[0], 0.0)
        self.assertAlmostEqual(root[3], 4.0)
        self.assertAlmostEqual(root[4], 4.0)

    def test_leaves_have_no_children(self):
        """All child indices of leaf nodes must be -1."""
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(2), Point(lat=0, lon=0, alt=0))
        # For a depth-2 tree the leaves are nodes 1–4
        for node_idx in range(1, 5):
            with self.subTest(node=node_idx):
                self.assertTrue(np.all(tree._children[node_idx] == -1))

    def test_root_has_four_children(self):
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(2), Point(lat=0, lon=0, alt=0))
        self.assertTrue(np.all(tree._children[0] == [1, 2, 3, 4]))

    def test_parent_aabb_contains_children_aabbs(self):
        """Every internal node AABB must contain each of its children's AABBs."""
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(4), Point(lat=0, lon=0, alt=0))
        data = tree._data
        children = tree._children
        for node_idx in range(len(data)):
            for c in children[node_idx]:
                if c == -1:
                    continue
                with self.subTest(parent=node_idx, child=int(c)):
                    self.assertTrue(
                        np.all(data[node_idx, :3] <= data[c, :3] + 1e-12),
                        msg=f"Node {node_idx} min exceeds child {c} min",
                    )
                    self.assertTrue(
                        np.all(data[node_idx, 3:] >= data[c, 3:] - 1e-12),
                        msg=f"Node {node_idx} max less than child {c} max",
                    )


# ===========================================================================
# 5. Line-of-sight query tests
# ===========================================================================


class TestHasLineOfSightClear(unittest.TestCase):
    """Rays that should pass through unobstructed."""

    def setUp(self):
        self.tree = _simple_2x2_tree()

    def test_ray_above_terrain(self):
        """Ray well above terrain (z > 1) travels horizontally → clear."""
        ray = Ray([0.5, 0.5, 2.0], [1, 0, 0], t_max=5.0)
        self.assertTrue(self.tree.has_line_of_sight(ray))

    def test_ray_beside_terrain_in_x(self):
        """Ray travels in +y at x = -1, entirely outside [0,2] → clear."""
        ray = Ray([-1, 0, 0.5], [0, 1, 0], t_max=3.0)
        self.assertTrue(self.tree.has_line_of_sight(ray))

    def test_ray_t_max_zero_outside_terrain(self):
        """t_max = 0 with origin above terrain → clear (ray travels nowhere)."""
        ray = Ray([1.0, 1.0, 2.0], [0, 0, 1], t_max=0.0)
        self.assertTrue(self.tree.has_line_of_sight(ray))

    def test_ray_too_short_to_reach_terrain(self):
        """Ray aimed at terrain but t_max too small to arrive → clear."""
        # Terrain top at z=1, origin at z=5, t_max=2 → ray stops at z=3
        ray = Ray([1.0, 1.0, 5.0], [0, 0, -1], t_max=2.0)
        self.assertTrue(self.tree.has_line_of_sight(ray))

    def test_ray_travelling_away_from_terrain(self):
        """Ray starts above terrain and moves further away → clear."""
        ray = Ray([1.0, 1.0, 2.0], [0, 0, 1], t_max=5.0)
        self.assertTrue(self.tree.has_line_of_sight(ray))


class TestHasLineOfSightBlocked(unittest.TestCase):
    """Rays that should be obstructed by terrain."""

    def setUp(self):
        self.tree = _simple_2x2_tree()

    def test_vertical_ray_through_centre(self):
        """Vertical ray downward from above hits terrain → blocked."""
        ray = Ray([1.0, 1.0, 2.0], [0, 0, -1], t_max=5.0)
        self.assertFalse(self.tree.has_line_of_sight(ray))

    def test_horizontal_ray_through_terrain(self):
        """Horizontal ray at mid-height sweeps through the terrain slab → blocked."""
        ray = Ray([-0.5, 1.0, 0.5], [1, 0, 0], t_max=10.0)
        self.assertFalse(self.tree.has_line_of_sight(ray))

    def test_diagonal_ray_enters_terrain(self):
        """45° diagonal ray enters a leaf AABB → blocked."""
        ray = Ray([-1, -1, 0.5], [1, 1, 0], t_max=10.0)
        self.assertFalse(self.tree.has_line_of_sight(ray))

    def test_ray_just_long_enough_to_reach_terrain(self):
        """t_max sufficient to enter terrain → blocked."""
        # Origin z=2, direction -z; terrain top at z=1, so t=1 enters terrain
        ray = Ray([1.0, 1.0, 2.0], [0, 0, -1], t_max=1.5)
        self.assertFalse(self.tree.has_line_of_sight(ray))

    def test_origin_inside_terrain(self):
        """Ray origin inside a leaf AABB, t_max = 0 → blocked."""
        ray = Ray([1.0, 1.0, 0.5], [0, 0, 1], t_max=0.0)
        self.assertFalse(self.tree.has_line_of_sight(ray))


class TestHasLineOfSightEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tree = _simple_2x2_tree()

    def test_result_is_bool(self):
        """Return type must be bool for both clear and blocked rays."""
        clear_ray = Ray([0, 0, 5], [0, 0, 1], t_max=1.0)
        blocked_ray = Ray([0, 0, 5], [0, 0, -1], t_max=10.0)
        self.assertIsInstance(self.tree.has_line_of_sight(clear_ray), bool)
        self.assertIsInstance(self.tree.has_line_of_sight(blocked_ray), bool)

    def test_ray_grazing_aabb_top_face(self):
        """Ray at exactly z=1 (top face of terrain) must not raise."""
        ray = Ray([1.0, 1.0, 1.0], [1, 0, 0], t_max=5.0)
        result = self.tree.has_line_of_sight(ray)
        self.assertIsInstance(result, bool)
        self.assertFalse(math.isinf(result))

    def test_repeated_queries_are_consistent(self):
        """Same ray queried multiple times must return the same result."""
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(4), Point(lat=0, lon=0, alt=0))
        blocked_ray = Ray([2.0, 2.0, 5.0], [0, 0, -1], t_max=10.0)
        clear_ray = Ray([2.0, 2.0, 5.0], [0, 0, 1], t_max=3.0)
        for _ in range(10):
            self.assertFalse(tree.has_line_of_sight(blocked_ray))
            self.assertTrue(tree.has_line_of_sight(clear_ray))


class TestHasLineOfSightDeepTrees(unittest.TestCase):
    """Line-of-sight queries on deeper (4×4 and 8×8) trees."""

    def test_4x4_clear(self):
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(4), Point(lat=0, lon=0, alt=0))
        ray = Ray([2.0, 2.0, 5.0], [0, 0, 1], t_max=3.0)
        self.assertTrue(tree.has_line_of_sight(ray))

    def test_4x4_blocked(self):
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(4), Point(lat=0, lon=0, alt=0))
        ray = Ray([2.0, 2.0, 5.0], [0, 0, -1], t_max=10.0)
        self.assertFalse(tree.has_line_of_sight(ray))

    def test_8x8_clear(self):
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(8), Point(lat=0, lon=0, alt=0))
        ray = Ray([4.0, 4.0, 5.0], [0, 0, 1], t_max=3.0)
        self.assertTrue(tree.has_line_of_sight(ray))

    def test_8x8_blocked(self):
        tree = HbvTree.from_leaf_bboxes(_unit_leaf_grid(8), Point(lat=0, lon=0, alt=0))
        ray = Ray([4.0, 4.0, 5.0], [0, 0, -1], t_max=10.0)
        self.assertFalse(tree.has_line_of_sight(ray))


# ===========================================================================
# 6. Persistence (save / load round-trip)
# ===========================================================================


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.tree = HbvTree.from_leaf_bboxes(
            _unit_leaf_grid(4), Point(lat=0, lon=0, alt=0)
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        self.path = tmp.name
        self.tree.save(self.path)
        self.loaded = HbvTree.load(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_data_array_preserved(self):
        np.testing.assert_array_equal(self.tree._data, self.loaded._data)

    def test_children_array_preserved(self):
        np.testing.assert_array_equal(self.tree._children, self.loaded._children)

    def test_stack_size_preserved(self):
        self.assertEqual(self.tree._stack_size, self.loaded._stack_size)

    def test_queries_match_original(self):
        """Reloaded tree must give identical query results to the original."""
        rays = [
            Ray([2, 2, 5], [0, 0, -1], 10.0),  # blocked
            Ray([2, 2, 5], [0, 0, 1], 3.0),  # clear
            Ray([-1, 2, 0.5], [1, 0, 0], 10.0),  # blocked (enters terrain)
        ]
        for ray in rays:
            with self.subTest(ray_origin=ray.p_start):
                self.assertEqual(
                    self.tree.has_line_of_sight(ray),
                    self.loaded.has_line_of_sight(ray),
                )


class TerrainTest(unittest.TestCase):
    def test_real_case(self):
        srtm = SrtmTerrainModel()
        tree = HbvTree.load(
            f"{TERRAIN_HBV_DATA_DIR}/tree_lat46:47_lon7:9_subsamplestride2.zip"
        )
        # tree = build_terrain_tree(46, 47, 7, 9)
        # tree.save("tree_lat46:47_lon7:9.zip")
        fast = FastSrtmModel(tree=tree, srtm_model=srtm)

        p1 = Point(lat=46.55, lon=8.0, alt=4000)
        p2 = Point(lat=46.55, lon=8.6, alt=1500)
        self.assertFalse(fast.has_line_of_sight(p1, p2))


if __name__ == "__main__":
    unittest.main()


# class FastLosTest(unittest.TestCase):
#     """
#     Important tests:
#     ----------------

#     - HbvTree.has_line_of_sight()
#     """


# class BuildBoundingBoxTest(unittest.TestCase):


if __name__ == "__main__":
    unittest.main()
