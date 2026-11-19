import datetime
import unittest

import numpy as np

from theia.detection.ecef_sampler import EcefDetectionSampler
from theia.types import Point

T = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
POINT = Point(lat=47.0, lon=8.0, alt=500.0)


class EcefDetectionSamplerTest(unittest.TestCase):
    def test_noise_covar_holds_variances(self):
        """
        The measurement model's noise covariance must hold the variances, not
        the standard deviations, of the sampled position.
        """
        sampler = EcefDetectionSampler(np.random.default_rng(0))
        detection = sampler._sample_position(POINT, (10.0, 20.0, 30.0), T)
        np.testing.assert_array_equal(
            detection.measurement_model.noise_covar,
            np.diag([100.0, 400.0, 900.0]),
        )

    def test_noise_covar_matches_sample_spread(self):
        """
        The covariance reported to the tracker must describe the actual spread
        of the sampled positions. Otherwise, the tracker is over- or
        under-confident.
        """
        sampler = EcefDetectionSampler(np.random.default_rng(0))
        sigmas = (1000.0, 2000.0, 3000.0)
        detections = [sampler._sample_position(POINT, sigmas, T) for _ in range(5000)]
        samples = np.array([np.array(d.state_vector).flatten() for d in detections])
        empirical_variances = np.var(samples, axis=0)
        np.testing.assert_allclose(
            empirical_variances,
            np.diag(detections[0].measurement_model.noise_covar),
            rtol=0.1,
        )


if __name__ == "__main__":
    unittest.main()
