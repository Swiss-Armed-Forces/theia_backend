import itertools
import unittest

import numpy as np
from scipy.stats import chi2

from theia.ellipsoid import Ellipsoid


class EllipsoidTest(unittest.TestCase):
    def setUp(self):
        N = 100
        rng = np.random.Generator(np.random.PCG64(seed=4287229932))
        p1_vec = rng.uniform(-100, 100, (N, 3))
        p2_vec = rng.uniform(-100, 100, (N, 3))
        ellipsoids: list[Ellipsoid] = []
        for p1, p2 in zip(p1_vec, p2_vec, strict=True):
            r = rng.uniform(np.linalg.norm(p1 - p2), 1000)
            ellipsoids.append(Ellipsoid(p1=tuple(p1), p2=tuple(p2), r=r))
        self._ellipsoids = ellipsoids

    def test_center(self):
        for ellipsoid in self._ellipsoids:
            expected = (np.array(ellipsoid.p1) + np.array(ellipsoid.p2)) / 2.0
            self.assertTrue(np.allclose(ellipsoid.center, expected))

    def test_axes_directions(self):
        for ellipsoid in self._ellipsoids:
            axes = ellipsoid.axes_directions

            # Test orthonormality.
            for ax in axes:
                self.assertAlmostEqual(np.linalg.norm(ax), 1)
            for ax1, ax2 in itertools.combinations(axes, 2):
                self.assertAlmostEqual(np.dot(ax1, ax2), 0.0)

            # Test alignment with focal points.
            focal_direction = np.array(ellipsoid.p1) - np.array(ellipsoid.p2)
            focal_direction = focal_direction / np.linalg.norm(focal_direction)
            alignments = [np.abs(np.dot(ax, focal_direction)) for ax in axes]
            alignments = sorted(alignments)
            self.assertTrue(np.isclose(alignments, (0, 0, 1)).all())

    def test_axes_lengths(self):
        for ellipsoid in self._ellipsoids:
            # Check that the lengths are correct.
            lengths = np.sort(ellipsoid.axes_lengths)
            d = np.linalg.norm(np.array(ellipsoid.p1) - np.array(ellipsoid.p2))
            a = ellipsoid.r / 2
            b = 0.5 * np.sqrt(ellipsoid.r**2 - d**2)
            c = b
            expected_lengths = np.sort(np.array((a, b, c)))
            self.assertTrue(np.isclose(lengths, expected_lengths).all())

            # Check that the order is consistent with the main semit-axes direction.
            for length, ax in zip(
                ellipsoid.axes_lengths, ellipsoid.axes_directions, strict=True
            ):
                p = ellipsoid.center + length * np.array(ax)
                # fmt: off
                radius = np.linalg.norm(ellipsoid.p1 - p) \
                    + np.linalg.norm(ellipsoid.p2 - p)
                # fmt: on
                self.assertAlmostEqual(radius, ellipsoid.r)
    
    def test_sampling(self):
        rng = np.random.default_rng(seed=4837)
        for ellipsoid in self._ellipsoids[:10]:
            result = chi_squared_uniformity_test(ellipsoid, rng, n_samples=50_000)
            assert not result["reject_null"], (
                f"Chi-squared test rejected uniform distribution "
                f"(focal_dist=?, r={ellipsoid.r}, p={result['p_value']:.4f})"
            )



# Code by Claude.
def _area_antideriv(u: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Antiderivative of the spheroid surface area element with respect to
    u = cos(theta), integrated over azimuth (up to the constant 2*pi*b).

    For a sphere (a == b), the integrand is constant (= a^2), so F(u) = a^2 * u.
    """
    c_sq = a**2 - b**2
    if c_sq < 1e-12:  # degenerate sphere
        return a**2 * u
    c = np.sqrt(c_sq)
    inner = np.clip(a**2 - c_sq * u**2, 0.0, None)  # numerical safety
    return 0.5 * u * np.sqrt(inner) + (a**2 / (2.0 * c)) * np.arcsin(
        np.clip(c * u / a, -1.0, 1.0)
    )


def chi_squared_uniformity_test(
    ellipsoid: Ellipsoid,
    rng: np.random.Generator,
    n_samples: int = 100_000,
    n_bins: int = 20,
    alpha: float = 0.05,
) -> dict:
    """
    Chi-squared goodness-of-fit test for uniform surface sampling on a
    prolate spheroid.

    Strategy
    --------
    Bin samples by u = cos(theta), the normalised projection onto the major
    axis.  For a surface-uniform distribution the expected count in each bin
    is proportional to the integral of the surface area element over that bin,
    computed analytically via _area_antideriv.

    Parameters
    ----------
    ellipsoid : Ellipsoid
    rng       : numpy random Generator
    n_samples : number of surface samples to draw
    n_bins    : number of equal-width bins in u = cos(theta) ∈ [-1, 1]
    alpha     : significance level for the reject/accept decision

    Returns
    -------
    dict with keys:
        statistic   chi-squared test statistic
        p_value     p-value (right tail)
        dof         degrees of freedom (n_bins - 1)
        reject_null True if p_value < alpha
        observed    array of observed counts per bin
        expected    array of expected counts per bin
    """
    a, b, _ = ellipsoid.axes_lengths
    ax1 = np.array(ellipsoid.axes_directions[0])
    center = np.array(ellipsoid.center)

    # --- Draw samples and project onto the major axis -------------------
    samples = np.array(
        [ellipsoid.sample_surface_uniformly(rng) for _ in range(n_samples)]
    )
    projections = (samples - center) @ ax1      # ∈ [-a, a]
    u = projections / a                          # cos(θ) ∈ [-1, 1]

    # --- Observed counts -------------------------------------------------
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    observed, _ = np.histogram(u, bins=bin_edges)

    # --- Expected counts (analytic area element) -------------------------
    F = _area_antideriv(bin_edges, a, b)
    expected_raw = np.diff(F)                    # proportional to area of each bin
    expected = expected_raw / expected_raw.sum() * n_samples

    # Guard against empty bins, which would make the statistic undefined.
    if np.any(expected < 5):
        raise ValueError(
            f"Some bins have expected count < 5 ({expected.min():.1f}). "
            "Reduce n_bins or increase n_samples."
        )

    # --- Chi-squared statistic ------------------------------------------
    stat = float(np.sum((observed - expected) ** 2 / expected))
    dof = n_bins - 1
    p_value = float(chi2.sf(stat, df=dof))

    return {
        "statistic": stat,
        "p_value": p_value,
        "dof": dof,
        "reject_null": p_value < alpha,
        "observed": observed,
        "expected": expected,
    }


if __name__ == "__main__":
    unittest.main()
