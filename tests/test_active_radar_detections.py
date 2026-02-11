# import unittest

# import numpy as np

# from theia.coordinates import CoordinateTransformations
# from theia.detection.active import calculate_monostatic_detection
# from theia.types import Point, Polarization, Radar, Receiver, Target, Transmitter


# class ActiveRadarDetectionTest(unittest.TestCase):
#     def test_runs(self):
#         p = Point(
#             lat=47.36700085728634,
#             lon=8.537724304199216,
#             alt=407.83600886023686,
#         )
#         ah = 10.0
#         bandwidth = 1
#         diameter = 2.0
#         radar = Radar(
#             transmitter=Transmitter(
#                 id=585,
#                 point=p,
#                 power=20000,
#                 erp=800,
#                 antenna_height=ah,
#                 antenna_diameter=diameter,
#                 frequency=1000.0,
#                 pulse_width=1,
#                 bandwidth=bandwidth,
#                 polarization=Polarization.HORIZONTAL,
#             ),
#             receiver=Receiver(
#                 id=585,
#                 point=p,
#                 antenna_height=ah,
#                 diameter=2.0,
#                 cpi_pulses=1,
#                 pfa=1e-6,
#                 min_elevation=-20.0,
#                 max_elevation=60.0,
#                 rotation_time=10.0,
#                 bandwidth=bandwidth,
#             ),
#         )

#         p = Point(
#             lat=47.367001,
#             lon=8.537724,
#             alt=2000,
#         )
#         target = Target(
#             id=0,
#             point=p,
#             cross_section=2.0,
#             velocity=CoordinateTransformations.velocity_geodetic_to_cartesian(
#                 p,
#                 vlat=100,
#                 vlon=0,
#                 valt=0,
#             ),
#         )

#         rng = np.random.Generator(np.random.PCG64(seed=4074992))
#         detection = calculate_monostatic_detection(radar, target, rng)

#         self.assertIsNotNone(detection)


# if __name__ == "__main__":
#     unittest.main()
