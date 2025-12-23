import unittest

from theia.doppler import monostatic_doppler


class MonostaticDopplerTest(unittest.TestCase):
    def test_monostatic_doppler(self):
        TOLERANCE = 0.6

        correct = -148.6210994231124
        calculated = monostatic_doppler(
            1000.0,
            47.36700085728634,
            8.537724304199216,
            407.83600886023686,
            47.348,
            8.6266,
            1000.0,
            250.0,
            0.0,
            0.0,
        )
        self.assertAlmostEqual(calculated, correct, delta=TOLERANCE)

        correct = 226.27107714677888
        calculated = monostatic_doppler(
            1000.0,
            47.36700085728634,
            8.537724304199216,
            407.83600886023686,
            47.406,
            8.3926,
            1000.0,
            250.0,
            0.0,
            0.0,
        )
        self.assertAlmostEqual(calculated, correct, delta=TOLERANCE)


if __name__ == "__main__":
    unittest.main()
