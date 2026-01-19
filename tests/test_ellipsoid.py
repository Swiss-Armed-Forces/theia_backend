import unittest

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.ellipsoid import Ellipsoid


class EllipsoidTest(unittest.TestCase):
    def test_consistency(self):
        n_examples = 10
        rng = np.random.Generator(np.random.PCG64(463799))
        all_p1 = rng.uniform(low=-1000, high=1000, size=((n_examples, 3)))
        all_p2 = rng.uniform(low=-1000, high=1000, size=((n_examples, 3)))
        bistatic_ranges = rng.uniform(
            low=np.linalg.norm(all_p1 - all_p2), high=100_000, size=(n_examples)
        )
        for p1, p2, r in zip(all_p1, all_p2, bistatic_ranges, strict=True):
            e = Ellipsoid(p1, p2, r)
            surface_points = e.sample_surface(n_theta=180, n_phi=180)
            for p in surface_points:
                self.assertTrue(
                    e.is_on_surface(
                        p,
                        point_in_world_coord=True,
                    )
                )

    def test_points_far_away(self):
        bound = 1000
        n_examples = 10

        rng = np.random.Generator(np.random.PCG64(463799))
        all_p1 = rng.uniform(low=-bound, high=bound, size=((n_examples, 3)))
        all_p2 = rng.uniform(low=-bound, high=bound, size=((n_examples, 3)))
        bistatic_ranges = rng.uniform(
            low=np.linalg.norm(all_p1 - all_p2), high=10 * bound, size=(n_examples)
        )
        for p1, p2, r in zip(all_p1, all_p2, bistatic_ranges):
            n_per_ellipse = 3
            e = Ellipsoid(p1, p2, r)
            x_min_ellipsoid = min(p1[0], p2[0]) - r
            x_max_ellipsoid = max(p1[0], p2[0]) + r
            y_min_ellipsoid = min(p1[1], p2[1]) - r
            y_max_ellipsoid = max(p1[1], p2[1]) + r
            z_min_ellipsoid = min(p1[2], p2[2]) - r
            z_max_ellipsoid = max(p1[2], p2[2]) + r

            n = 0
            while n < n_per_ellipse:
                # Sample points outside an upper bound of the bounding box
                # of the ellipsoid.
                x, y, z = rng.uniform(low=-100 * bound, high=100 * bound, size=(3,))
                if (
                    (x_min_ellipsoid <= x <= x_max_ellipsoid)
                    and (y_min_ellipsoid <= y <= y_max_ellipsoid)
                    and (z_min_ellipsoid <= z <= z_max_ellipsoid)
                ):
                    continue
                else:
                    self.assertFalse(e.is_on_surface((x, y, z)))
                    n += 1


if __name__ == "__main__":
    unittest.main()
