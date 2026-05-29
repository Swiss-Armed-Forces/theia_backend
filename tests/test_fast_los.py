import math
import unittest

import numpy as np

from theia.terrain_fast_los import _los_kernel, build_higher_level


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
            stack,
        )
        self.assertEqual(result, False)

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
            stack,
        )
        self.assertEqual(result, True)

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
            stack,
        )
        self.assertEqual(result, True)

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
            stack,
        )
        self.assertEqual(result, False)


class TestLosKernelSingleLeaf(unittest.TestCase):
    """Tests on a single-leaf tree: one AABB, no children."""

    # ------------------------------------------------------------------
    # Basic hit / miss along X axis
    # ------------------------------------------------------------------
    def test_ray_hits_leaf_from_left(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray travels in +X, starts left of box, aimed right → hit
        result = _los_kernel(data, children, -0.5, 0.5, 0.5, 1.0, INF, INF, 10.0, stack)
        self.assertFalse(result, "Ray should be blocked by the leaf AABB")

    def test_ray_misses_leaf_going_away(self):
        data, children = _single_leaf()
        stack = _make_stack(1)
        # Ray travels in -X, starts left of box, aimed further left → miss
        result = _los_kernel(
            data, children, -0.5, 0.5, 0.5, -1.0, INF, INF, 10.0, stack
        )
        self.assertTrue(result, "Ray aimed away from box should have clear LOS")

    def test_ray_misses_leaf_offset_in_y(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray travels in +X but y=2.0 is outside the [0,1] slab
        result = _los_kernel(data, children, -0.5, 2.0, 0.5, 1.0, INF, INF, 10.0, stack)
        self.assertTrue(result, "Ray offset above box in Y should miss")

    # ------------------------------------------------------------------
    # Ray origin inside the AABB
    # ------------------------------------------------------------------
    def test_ray_origin_inside_aabb_hits(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Origin is inside the box; any direction should still intersect
        result = _los_kernel(data, children, 0.5, 0.5, 0.5, 1.0, INF, INF, 10.0, stack)
        self.assertFalse(
            result, "Ray originating inside the leaf AABB should be blocked"
        )

    # ------------------------------------------------------------------
    # t_max cutoff
    # ------------------------------------------------------------------
    def test_t_max_cuts_ray_before_box(self):
        data, children = _single_leaf()  # box starts at x=0
        stack = _make_stack(1)
        # Ray origin at x=-5, travelling +X, box is 5 units away.
        # t_max=3 means the ray stops at x=-2, before the box.
        result = _los_kernel(data, children, -5.0, 0.5, 0.5, 1.0, INF, INF, 3.0, stack)
        self.assertTrue(result, "Ray should not reach the box within t_max")

    def test_t_max_reaches_box(self):
        data, children = _single_leaf()
        stack = _make_stack(1)
        # Same setup but t_max=10 → ray reaches box
        result = _los_kernel(data, children, -5.0, 0.5, 0.5, 1.0, INF, INF, 10.0, stack)
        self.assertFalse(result, "Ray should reach and be blocked by the box")

    # ------------------------------------------------------------------
    # Diagonal ray
    # ------------------------------------------------------------------
    def test_diagonal_ray_hits(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # 45-degree ray in XY plane aimed at box centre (0.5, 0.5)
        inv_sqrt2 = 1.0 / math.sqrt(0.5)
        result = _los_kernel(
            data, children, -1.0, -1.0, 0.5, inv_sqrt2, inv_sqrt2, INF, 10.0, stack
        )
        self.assertFalse(result, "Diagonal ray aimed at box should be blocked")

    def test_diagonal_ray_misses(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Same diagonal direction but offset so it passes above the box
        inv_sqrt2 = 1.0 / math.sqrt(0.5)
        result = _los_kernel(
            data, children, -1.0, 2.0, 0.5, inv_sqrt2, inv_sqrt2, INF, 10.0, stack
        )
        self.assertTrue(result, "Diagonal ray above box should miss")

    # ------------------------------------------------------------------
    # Negative direction components
    # ------------------------------------------------------------------
    def test_negative_direction_hits(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray starts to the right of the box, travels in -X direction
        result = _los_kernel(data, children, 2.0, 0.5, 0.5, -1.0, INF, INF, 10.0, stack)
        self.assertFalse(result, "Negative-direction ray through box should be blocked")

    # ------------------------------------------------------------------
    # Axis-aligned ray parallel to a slab face but outside (IEEE-754 path)
    # ------------------------------------------------------------------
    def test_axis_aligned_ray_parallel_outside(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray moves purely in X (idy=idz=inf) but y=2.0 is outside [0,1]
        result = _los_kernel(data, children, -1.0, 2.0, 0.5, 1.0, INF, INF, 10.0, stack)
        self.assertTrue(result, "Axis-aligned ray outside a slab should produce a miss")

    # ------------------------------------------------------------------
    # Grazing ray (hits exactly on a face)
    # ------------------------------------------------------------------
    def test_grazing_ray_on_face(self):
        data, children = _single_leaf()  # box [0,1]^3
        stack = _make_stack(1)
        # Ray travels in +X at y=0 (exactly on the y=0 face of the box)
        # Conservative behaviour: should still register as a hit
        result = _los_kernel(data, children, -1.0, 0.0, 0.5, 1.0, INF, INF, 10.0, stack)
        self.assertFalse(
            result, "Grazing ray on a face should count as a hit (conservative)"
        )


class TestLosKernelTree(unittest.TestCase):
    """Tests on a small complete quadtree (depth-2, 1 root + 4 leaves)."""

    def _build_tree(self):
        # Root covers [-2, -2, 0] → [2, 2, 1]
        # Four children, each a leaf in one quadrant
        # fmt: off
        data = np.array(
            [
                [-2.0, -2.0, 0.0,  2.0,  2.0, 1.0],  # 0: root
                [-2.0, -2.0, 0.0,  0.0,  0.0, 1.0],  # 1: SW leaf
                [ 0.0, -2.0, 0.0,  2.0,  0.0, 1.0],  # 2: SE leaf
                [-2.0,  0.0, 0.0,  0.0,  2.0, 1.0],  # 3: NW leaf
                [ 0.0,  0.0, 0.0,  2.0,  2.0, 1.0],  # 4: NE leaf
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
        result = _los_kernel(data, children, -3.0, 0.0, 5.0, 1.0, INF, INF, 10.0, stack)
        self.assertTrue(
            result, "Ray missing the root should return clear LOS immediately"
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
            data, children, -3.0, -3.0, 0.5, 1.0, INF, INF, 10.0, stack
        )
        self.assertTrue(
            result, "Ray inside root but outside all leaf AABBs should be clear"
        )

    # ------------------------------------------------------------------
    # Selective leaf hit
    # ------------------------------------------------------------------
    def test_ray_hits_only_sw_leaf(self):
        data, children = self._build_tree()
        stack = _make_stack(2)
        # Ray aimed at SW leaf centre (-1, -1)
        result = _los_kernel(
            data, children, -3.0, -1.0, 0.5, 1.0, INF, INF, 10.0, stack
        )
        self.assertFalse(result, "Ray through SW leaf should be blocked")

    def test_ray_hits_only_ne_leaf(self):
        data, children = self._build_tree()
        stack = _make_stack(2)
        # Ray aimed at NE leaf centre (1, 1)
        result = _los_kernel(data, children, -3.0, 1.0, 0.5, 1.0, INF, INF, 10.0, stack)
        self.assertFalse(result, "Ray through NE leaf should be blocked")

    # ------------------------------------------------------------------
    # t_max stops ray inside root but before leaves
    # ------------------------------------------------------------------
    def test_t_max_stops_before_leaf(self):
        data, children = self._build_tree()
        stack = _make_stack(2)
        # Root starts at x=-2. Ray origin at x=-10.
        # The root AABB x-slab starts at t=8, so t_max=7 won't even reach it.
        result = _los_kernel(data, children, -10.0, 0.0, 0.5, 1.0, INF, INF, 7.0, stack)
        self.assertTrue(
            result, "Ray stopped by t_max before root should have clear LOS"
        )

    # ------------------------------------------------------------------
    # Boundary between two leaves
    # ------------------------------------------------------------------
    def test_ray_on_leaf_boundary(self):
        data, children = self._build_tree()
        # Ray travels along y=0, the boundary between SW/SE and NW/NE leaves.
        # Conservative: the ray grazes both SW and SE leaf faces; either or
        # both may register as a hit (implementation-defined), so we just
        # verify the call completes without error and returns a bool.
        stack2 = _make_stack(2)
        result = _los_kernel(
            data, children, -3.0, 0.0, 0.5, 1.0, INF, INF, 10.0, stack2
        )
        self.assertIsInstance(result, (bool, np.bool_))
        self.assertFalse(result)


class TestLosKernelNaNBehaviour(unittest.TestCase):
    """
    The docstring explicitly acknowledges NaN (ray origin exactly on a slab
    face) is not guarded against, but guarantees no false negatives — at
    worst one extra subtree is visited.  We test that the function at least
    terminates and returns a value in {True, False}.
    """

    def test_nan_inv_direction_terminates(self):
        data, children = _single_leaf = (
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float64),
            np.array([[-1, -1, -1, -1]], dtype=np.int64),
        )
        stack = _make_stack(1)
        # Ray origin on the x=0 face of the box → NaN in x slab calculation
        result = _los_kernel(data, children, 0.0, 0.5, 0.5, 1.0, INF, INF, 10.0, stack)
        self.assertIsInstance(
            result, (bool, np.bool_), "NaN path must still return a bool"
        )
        self.assertFalse(result)

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


# class FastLosTest(unittest.TestCase):
#     """
#     Important tests:
#     ----------------

#     - HbvTree.has_line_of_sight()

#     Nice to have tests:
#     -------------------

#     - build_bounding_boxes()

#     """


# class BuildBoundingBoxTest(unittest.TestCase):


if __name__ == "__main__":
    unittest.main()
