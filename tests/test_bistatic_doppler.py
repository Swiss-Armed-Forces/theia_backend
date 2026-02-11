import unittest

from theia.data_loading_testing import load_pcl_reference_data
from theia.doppler import calculate_doppler_shift


class BistaticDopplerTest(unittest.TestCase):
    def test_bistatic_doppler_against_reference(self):
        _, detections = load_pcl_reference_data()
        self.assertGreater(len(detections), 0)
        for detection, snr in detections:
            doppler_ref = detection.doppler_shift
            doppler_calculated = calculate_doppler_shift(
                detection.receiver,
                detection.target,
                detection.transmitter,
            )
            self.assertAlmostEqual(doppler_ref, doppler_calculated, delta=0.21)

if __name__ == "__main__":
    unittest.main()
