import unittest

import numpy as np

from theia import elevationAt


class ElevationAtTest(unittest.TestCase):
    def test_elevationAt(self):
        # Tests.
        # Each row is another test, consisting of (lat, lon) and the expected height.
        # The points where chosen randomly using https://www.randomcoords.com/region/europe.
        # and the correct heights from https://latlongdata.com/elevation/.
        test_cases = np.array(
            [
                [47.1380402, 9.5184237, 453.86],
                [46.2313104, 7.3658079, 500.88],
                [45.7391548, 7.3046952, 604.31],
                [47.8007899, 13.0441258, 423.64],
                [49.7452357, 6.6357603, 137.52],
                [50.0973787, 14.5286722, 218.03],
            ]
        )

        for i in range(test_cases.shape[0]):
            lat, lon, correct_height = test_cases[i, :]
            height = elevationAt(lat, lon)
            diff = np.abs(height - correct_height)
            self.assertLess(diff, 10)


if __name__ == "__main__":
    unittest.main()
