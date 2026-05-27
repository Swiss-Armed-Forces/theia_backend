# ── Helpers ──────────────────────────────────────────────────────────────────

import unittest

from theia.terrain_fast_los import Node, Ray


def unit_box(offset: float = 0.0) -> Node:
    """A 1×1×1 leaf node whose near corner is at (offset, offset, offset)."""
    o = offset
    return Node(
        x_min=o,
        x_max=o + 1,
        y_min=o,
        y_max=o + 1,
        z_min=o,
        z_max=o + 1,
        children=None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Ray dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestRay(unittest.TestCase):
    def test_fields_stored(self):
        r = Ray(p_start=(1.0, 2.0, 3.0), direction=(0.0, 0.0, 1.0), t_max=100.0)
        self.assertEqual(r.p_start, (1.0, 2.0, 3.0))
        self.assertEqual(r.direction, (0.0, 0.0, 1.0))
        self.assertEqual(r.t_max, 100.0)

    def test_equality(self):
        r1 = Ray((0, 0, 0), (1, 0, 0), 50)
        r2 = Ray((0, 0, 0), (1, 0, 0), 50)
        self.assertEqual(r1, r2)

    def test_inequality(self):
        r1 = Ray((0, 0, 0), (1, 0, 0), 50)
        r2 = Ray((0, 0, 0), (0, 1, 0), 50)
        self.assertNotEqual(r1, r2)


# ═══════════════════════════════════════════════════════════════════════════
# Node.is_leaf
# ═══════════════════════════════════════════════════════════════════════════


class TestIsLeaf(unittest.TestCase):
    def test_leaf_when_children_none(self):
        self.assertTrue(unit_box().is_leaf())

    def test_not_leaf_when_children_set(self):
        children = (unit_box(), unit_box(), unit_box(), unit_box())
        node = Node(0, 2, 0, 2, 0, 2, children=children)
        self.assertFalse(node.is_leaf())


# ═══════════════════════════════════════════════════════════════════════════
# Node._intersect_current_level  (slab-test geometry)
# ═══════════════════════════════════════════════════════════════════════════


class TestIntersectCurrentLevel(unittest.TestCase):
    """Tests for the AABB slab-intersection helper."""

    # ------------------------------------------------------------------ hits

    def test_hit_along_x_axis(self):
        """Ray approaching head-on along x enters at x_min."""
        node = unit_box()
        # x_min=0; start=-1 → t_enter = (0 - (-1)) / 1 = 1
        ray = Ray(p_start=(-1.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        t = node._intersect_current_level(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 1.0)

    def test_hit_along_y_axis(self):
        """Ray approaching head-on along y enters at y_min."""
        node = unit_box()
        # y_min=0; start=-2 → t_enter = 2
        ray = Ray(p_start=(0.5, -2.0, 0.5), direction=(0.0, 1.0, 0.0), t_max=10)
        t = node._intersect_current_level(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 2.0)

    def test_hit_along_z_axis(self):
        """Ray approaching head-on along z enters at z_min."""
        node = unit_box()
        ray = Ray(p_start=(0.5, 0.5, -3.0), direction=(0.0, 0.0, 1.0), t_max=10)
        t = node._intersect_current_level(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 3.0)

    def test_hit_diagonal_xy(self):
        """45° ray in the XY plane entering the [0,1]³ box at its x_min/y_min corner."""
        node = unit_box()
        # Both x and y slabs enter at t=1; z is parallel and 0.5 is inside [0,1].
        ray = Ray(p_start=(-1.0, -1.0, 0.5), direction=(1.0, 1.0, 0.0), t_max=10)
        t = node._intersect_current_level(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 1.0)

    def test_hit_returns_entry_t_not_exit_t(self):
        """Entry t (x_min face) must be returned, not the exit t (x_max face)."""
        node = unit_box()
        # start=-2; x_min=0 → t_enter=2; x_max=1 → t_exit=3
        ray = Ray(p_start=(-2.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        t = node._intersect_current_level(ray)
        self.assertAlmostEqual(t, 2.0)

    def test_hit_negative_direction_hits_x_max_face(self):
        """Ray travelling in -x enters via x_max."""
        node = unit_box()
        # (x_max=1 - start=3) / d=-1 → t=2
        ray = Ray(p_start=(3.0, 0.5, 0.5), direction=(-1.0, 0.0, 0.0), t_max=10)
        t = node._intersect_current_level(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 2.0)

    # ----------------------------------------------------------------- misses

    def test_miss_ray_pointing_away(self):
        """Ray starts beyond x_max and travels away; all x-slab ts are negative."""
        node = unit_box()
        ray = Ray(p_start=(2.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertIsNone(node._intersect_current_level(ray))

    def test_miss_parallel_ray_above_y_slab(self):
        """Ray travels along x but its y origin (5.0) is outside [y_min=0, y_max=1]."""
        node = unit_box()
        ray = Ray(p_start=(-1.0, 5.0, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertIsNone(node._intersect_current_level(ray))

    def test_miss_parallel_ray_below_z_slab(self):
        """Ray travels along x but its z origin (-1.0) is outside [z_min=0, z_max=1]."""
        node = unit_box()
        ray = Ray(p_start=(-1.0, 0.5, -1.0), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertIsNone(node._intersect_current_level(ray))

    def test_miss_slab_intervals_do_not_overlap(self):
        """
        Diagonal ray where x and y slab intervals are disjoint.
        Thin box: x=[0,0.1], y=[0,1], z=[0,1].
        Ray start=(-1,-2,0.5), direction=(1,1,0):
          x-slab: t_enter=1.0, t_exit=1.1
          y-slab: t_enter=2.0, t_exit=3.0
          max(t_enter)=2.0 > min(t_exit)=1.1 → miss.
        """
        node = Node(
            x_min=0.0,
            x_max=0.1,
            y_min=0.0,
            y_max=1.0,
            z_min=0.0,
            z_max=1.0,
            children=None,
        )
        ray = Ray(p_start=(-1.0, -2.0, 0.5), direction=(1.0, 1.0, 0.0), t_max=10)
        self.assertIsNone(node._intersect_current_level(ray))

    def test_miss_zero_direction_origin_outside_slab(self):
        """All-zero direction with origin outside every slab → miss."""
        node = unit_box()
        ray = Ray(p_start=(-1.0, -1.0, -1.0), direction=(0.0, 0.0, 0.0), t_max=10)
        self.assertIsNone(node._intersect_current_level(ray))

    def test_miss_zero_direction_x_origin_inside_but_y_outside(self):
        """Direction zero; x and z origins are inside their slabs but y is not."""
        node = unit_box()
        ray = Ray(p_start=(0.5, 5.0, 0.5), direction=(0.0, 0.0, 0.0), t_max=10)
        self.assertIsNone(node._intersect_current_level(ray))

    # --------------------------------------------------------- edge / boundary

    def test_ray_starting_on_entry_face(self):
        """Origin sits exactly on x_min; t_enter=0 should be returned."""
        node = unit_box()
        ray = Ray(p_start=(0.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        t = node._intersect_current_level(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 0.0)

    def test_ray_starting_inside_returns_exit_t(self):
        """
        Origin is strictly inside the box; t_enter is negative so the method
        must return t_exit (the exit face), not t_enter.
        """
        node = unit_box()
        # x_enter=(0-0.5)/1=-0.5 (<0), x_exit=(1-0.5)/1=0.5
        ray = Ray(p_start=(0.5, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        t = node._intersect_current_level(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Node.intersect  (public recursive API)
# ═══════════════════════════════════════════════════════════════════════════


class TestIntersect(unittest.TestCase):
    """Tests for the recursive public intersect method."""

    # ----------------------------------------------------------------- leaves

    def test_leaf_hit_returns_t(self):
        node = unit_box()
        ray = Ray(p_start=(-1.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertAlmostEqual(node.intersect(ray), 1.0)

    def test_leaf_miss_returns_none(self):
        node = unit_box()
        ray = Ray(p_start=(5.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertIsNone(node.intersect(ray))

    def test_leaf_parallel_ray_outside_slab_returns_none(self):
        """With the slab test a parallel ray that misses a slab must be None."""
        node = unit_box()
        ray = Ray(p_start=(-1.0, 5.0, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertIsNone(node.intersect(ray))

    # -------------------------------------------------- internal nodes / trees

    def _four_collinear_children(self) -> Node:
        """
        Four children along the x-axis: [0,1]³, [2,3]³, [4,5]³, [6,7]³,
        all at y=z=0..1.  Parent encloses them all.
        A ray at y=0.5, z=0.5 hits them in order at t=1, 3, 5, 7.
        """
        c0 = unit_box(0.0)
        c1 = unit_box(2.0)
        c2 = unit_box(4.0)
        c3 = unit_box(6.0)
        return Node(
            x_min=0,
            x_max=7,
            y_min=0,
            y_max=1,
            z_min=0,
            z_max=1,
            children=(c0, c1, c2, c3),
        )

    def test_internal_node_returns_closest_child_t(self):
        parent = self._four_collinear_children()
        ray = Ray(p_start=(-1.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=20)
        self.assertAlmostEqual(parent.intersect(ray), 1.0)

    def test_internal_node_miss_returns_none(self):
        parent = self._four_collinear_children()
        ray = Ray(p_start=(-1.0, 0.5, 0.5), direction=(-1.0, 0.0, 0.0), t_max=20)
        self.assertIsNone(parent.intersect(ray))

    def test_parent_miss_short_circuits_children(self):
        """If the parent AABB is missed, no child is evaluated."""
        parent = self._four_collinear_children()
        # Ray parallel to x but y=5 is outside the parent's y-slab [0,1].
        ray = Ray(p_start=(-1.0, 5.0, 0.5), direction=(1.0, 0.0, 0.0), t_max=20)
        self.assertIsNone(parent.intersect(ray))

    def test_all_children_missed_returns_none(self):
        """
        Parent is hit but every child is missed because the ray's y=5 is outside
        each child's y-slab [0,1].  The parent's y_max=10 is deliberately larger
        so the ray hits the parent but none of the children.
        """
        c0 = Node(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, children=None)
        c1 = Node(x_min=2, x_max=3, y_min=0, y_max=1, z_min=0, z_max=1, children=None)
        c2 = Node(x_min=4, x_max=5, y_min=0, y_max=1, z_min=0, z_max=1, children=None)
        c3 = Node(x_min=6, x_max=7, y_min=0, y_max=1, z_min=0, z_max=1, children=None)
        parent = Node(
            x_min=0,
            x_max=7,
            y_min=0,
            y_max=10,
            z_min=0,
            z_max=1,
            children=(c0, c1, c2, c3),
        )
        ray = Ray(p_start=(-1.0, 5.0, 0.5), direction=(1.0, 0.0, 0.0), t_max=20)
        self.assertIsNone(parent.intersect(ray))

    def test_two_level_tree_hit_propagates(self):
        """grandparent → parent → leaf; the correct leaf t must bubble up."""
        leaf = unit_box()  # [0,1]³
        mid = Node(
            x_min=0,
            x_max=1,
            y_min=0,
            y_max=1,
            z_min=0,
            z_max=1,
            children=(leaf, unit_box(10), unit_box(20), unit_box(30)),
        )
        root = Node(
            x_min=-1,
            x_max=11,
            y_min=-1,
            y_max=11,
            z_min=-1,
            z_max=11,
            children=(mid, unit_box(5), unit_box(6), unit_box(7)),
        )
        ray = Ray(p_start=(-0.5, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=20)
        t = root.intersect(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 0.5)  # entry into leaf at x_min=0

    def test_two_level_tree_miss_returns_none(self):
        leaf = unit_box()
        mid = Node(
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            z_min=0.0,
            z_max=1.0,
            children=(leaf, unit_box(10.0), unit_box(20.0), unit_box(30.0)),
        )
        root = Node(
            x_min=-1.0,
            x_max=11.0,
            y_min=-1.0,
            y_max=11.0,
            z_min=-1.0,
            z_max=11.0,
            children=(mid, unit_box(5.0), unit_box(6.0), unit_box(7.0)),
        )
        # y=50 is outside all AABBs
        ray = Ray(p_start=(-1.0, 50.0, 0.5), direction=(1.0, 0.0, 0.0), t_max=20)
        self.assertIsNone(root.intersect(ray))

    # ------------------------------------------------- return-type contract

    def test_hit_returns_float(self):
        node = unit_box()
        ray = Ray(p_start=(-1.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertIsInstance(node.intersect(ray), float)

    def test_miss_returns_none(self):
        node = unit_box()
        ray = Ray(p_start=(5.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertIsNone(node.intersect(ray))

    def test_returned_t_is_non_negative(self):
        node = unit_box()
        ray = Ray(p_start=(-1.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertGreaterEqual(node.intersect(ray), 0)


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases(unittest.TestCase):
    def test_ray_inside_box_returns_exit_t(self):
        """Ray origin inside the box → return the exit face distance."""
        node = unit_box()
        # x_enter=(0-0.5)/1=-0.5 → negative; x_exit=(1-0.5)/1=0.5
        ray = Ray(p_start=(0.5, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertAlmostEqual(node.intersect(ray), 0.5)

    def test_large_ecef_coordinate_values(self):
        """ECEF coordinates are ~6×10⁶ m; verify no numerical issues."""
        R = 6_378_137.0
        node = Node(
            x_min=R - 1000,
            x_max=R,
            y_min=-500.0,
            y_max=500.0,
            z_min=-500.0,
            z_max=500.0,
            children=None,
        )
        ray = Ray(p_start=(R - 5000, 0.0, 0.0), direction=(1.0, 0.0, 0.0), t_max=1e9)
        t = node.intersect(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 4000.0, places=3)

    def test_unit_direction_t_equals_euclidean_distance(self):
        """With a unit direction vector t is the Euclidean distance to the surface."""
        node = unit_box()
        # 5 m from x_min along the x-axis
        ray = Ray(p_start=(-5.0, 0.5, 0.5), direction=(1.0, 0.0, 0.0), t_max=20)
        self.assertAlmostEqual(node.intersect(ray), 5.0)

    def test_grazing_ray_along_face(self):
        """Ray travelling exactly along a face (y=0, which equals y_min) must hit."""
        node = unit_box()
        ray = Ray(p_start=(-1.0, 0.0, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        t = node.intersect(ray)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 1.0)

    def test_parallel_ray_exactly_on_slab_boundary_hits(self):
        """A parallel ray with origin on the slab boundary (y == y_min) is a hit."""
        node = unit_box()
        ray = Ray(p_start=(-1.0, 0.0, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertIsNotNone(node.intersect(ray))

    def test_parallel_ray_just_outside_slab_misses(self):
        """A parallel ray with origin just outside the slab must miss."""
        node = unit_box()
        ray = Ray(p_start=(-1.0, -1e-9, 0.5), direction=(1.0, 0.0, 0.0), t_max=10)
        self.assertIsNone(node.intersect(ray))


if __name__ == "__main__":
    unittest.main()
