import unittest

from theia.data_loading_testing import load_pcl_reference_data
from theia.doppler import calculate_bistatic_doppler


class BistaticDopplerTest(unittest.TestCase):
    def test_bistatic_doppler_against_reference(self):
        detections = load_pcl_reference_data()
        for detection in detections:
            doppler_ref = detection.doppler_shift
            doppler_calculated = calculate_bistatic_doppler(
                detection.receiver,
                detection.target,
                detection.transmitter,
            )
            self.assertLess(
                abs(doppler_ref - doppler_calculated) / doppler_ref * 100,
                0.3,
            )


if __name__ == "__main__":
    unittest.main()
