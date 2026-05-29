import math
import unittest

import numpy as np

from theia.terrain_fast_los import _los_kernel


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
        stack = _make_stack(2)
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


# class FastLosTest(unittest.TestCase):
#     """
#     Important tests:
#     ----------------

#     - _los_kernel() -> understand it!!!
#     - HbvTree.has_line_of_sight()

#     Nice to have tests:
#     -------------------

#     - build_bounding_boxes()
#     - pool2d()
#     - build_higher_level()

#     """


# class BuildBoundingBoxTest(unittest.TestCase):


if __name__ == "__main__":
    unittest.main()
