import unittest

from theia.data_loading_testing import load_pcl_reference_data
from theia.detection.passive import calculate_bistatic_detection, calculate_snr


class BistaticDetectionTest(unittest.TestCase):
    def test_against_reference(self):
        trajectories, detections = load_pcl_reference_data()
        for detection_ref, snr in detections:
            # print("ID = ", detection_ref.detection_id)
            bistatic_range_ref = detection_ref.bistatic_range
            bistatic_doppler_ref = detection_ref.doppler_shift
            detection = calculate_bistatic_detection(
                detection_ref.receiver,
                detection_ref.transmitter,
                detection_ref.target,
            )
            # snr_calc = calculate_snr(
            #     detection_ref.transmitter,
            #     detection_ref.receiver,
            #     detection_ref.target.point,

            # )

            self.assertIsNot(detection, None)

            self.assertAlmostEqual(bistatic_range_ref, detection.bistatic_range)
            self.assertAlmostEqual(
                bistatic_doppler_ref,
                detection.doppler_shift,
                delta=0.2,
            )
            # self.assertAlmostEqual(snr, snr_calc)


if __name__ == "__main__":
    unittest.main()
