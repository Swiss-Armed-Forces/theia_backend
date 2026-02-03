# import unittest

# import numpy as np
# from theia.detection.active import radar_detection_given_with_splat


# class ActiveRadarDetectionTest(unittest.TestCase):

#     def test_active_radar_detections(self):
#         line_of_sight_ok = 1.0
#         propagation_loss = 109.38008880615234
#         free_space_loss = 109.42449188232422
#         # fresnel_zone_free = 7044.87939453125
#         fresnel_zone_free = 0.0
#         target_radiation_distance = 7.064422898951063
#         result = radar_detection_given_with_splat(
#             20000,
#             2.0,
#             1000.0,
#             1.0,
#             1,
#             1,
#             1e-06,
#             1.0,
#             target_radiation_distance,
#         )
#         self.assertAlmostEqual(result, 1.0)

#         line_of_sight_ok = 1.0
#         propagation_loss = np.nan
#         free_space_loss = 113.8821029663086
#         # fresnel_zone_free = 11769.3916015625
#         fresnel_zone_free = 0.0
#         target_radiation_distance = 11.800535396299894
#         result = radar_detection_given_with_splat(
#             20000,
#             2.0,
#             1000.0,
#             1.0,
#             1,
#             1,
#             1e-06,
#             1.0,
#             target_radiation_distance,
#         )
#         self.assertAlmostEqual(result, 1.0)


# if __name__ == "__main__":
#     unittest.main()
