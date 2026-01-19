import unittest

import numpy as np

from theia.ellipsoid import Ellipsoid


class EllipsoidTest(unittest.TestCase):
    TOLERANCE = 30.0

    def test_consistency(self):
        n_examples = 10
        rng = np.random.Generator(np.random.PCG64(463799))
        p1 = rng.uniform(low=-1000, high=1000, size=((n_examples, 3)))
        p2 = rng.uniform(low=-1000, high=1000, size=((n_examples, 3)))
        bistatic_ranges = rng.uniform(low=np.linalg.norm(p2 - p1), high=100_000, size=(n_examples))
        for p1, p2, r in zip(p1, p2, bistatic_ranges, strict=True):
            e = Ellipsoid(p1, p2, r)
            surface_points = e.sample_surface(n_theta=180, n_phi=180)
            for p in surface_points:
                self.assertTrue(e.is_on_surface(p, tol=self.TOLERANCE))


if __name__ == "__main__":
    unittest.main()
