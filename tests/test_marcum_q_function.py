import unittest

import numpy as np

from theia.util import marcum_q_function


def _noise_only_marcum_q_function(beta: float) -> float:
    return np.exp(-(beta**2) / 2)


class MarcumQFunctionTest(unittest.TestCase):
    def test_normalisation(self):
        N = 1000
        rng = np.random.Generator(np.random.PCG64(seed=409367403745074))
        for alpha in rng.uniform(low=0, high=30, size=N):
            result = marcum_q_function(alpha, 0.0)
            self.assertAlmostEqual(result, 1)

    def test_noise_only(self):
        betas = np.linspace(0.0, 100, 1000)
        y_calc = [marcum_q_function(0.0, beta) for beta in betas]
        y_true = [_noise_only_marcum_q_function(beta) for beta in betas]
        for yc, yt in zip(y_calc, y_true, strict=True):
            self.assertAlmostEqual(yc, yt)

    def test_bounds(self):
        N = 1000
        rng = np.random.Generator(np.random.PCG64(seed=409367403745074))
        alphas = rng.uniform(low=0, high=30, size=N)
        betas = rng.uniform(low=0, high=10, size=N)

        # Test that the values of the marcum q function are in [0, 1].
        for alpha, beta in zip(alphas, betas, strict=True):
            y = marcum_q_function(alpha, beta)
            self.assertTrue(-1e-14 <= y <= 1)

    def test_zero(self):
        self.assertAlmostEqual(marcum_q_function(0.0, 0.0), 1.0)

    def test_against_reference(self):
        alphas = [
            26.47244252,
            10.04522928,
            5.02628013,
            2.4841613,
            18.25489627,
            19.22397514,
            5.8223934,
            26.13640425,
            16.77923787,
            27.05366587,
        ]
        betas = [
            8.36881916,
            3.5307805,
            2.79167796,
            4.40254198,
            0.32630529,
            4.64734232,
            1.42038438,
            6.48161315,
            7.56938362,
            7.8617206,
        ]

        # Calculated using Octave v9.3.0 (OctaveOnline)
        y_refs = [
            1.000000000000000,
            0.999999999978754,
            0.991055446383506,
            0.038611289266728,
            1.000000000000000,
            1.000000000000000,
            0.999997515252431,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
        ]

        for alpha, beta, y in zip(alphas, betas, y_refs, strict=True):
            self.assertAlmostEqual(marcum_q_function(alpha, beta), y)


if __name__ == "__main__":
    unittest.main()
