import unittest

from theia.util import erp_to_power

class UtilTest(unittest.TestCase):

    def test_erp_to_power(self):
        self.assertEqual(
            erp_to_power(30., 0., 0.),
            1000.,
        )
        self.assertTrue(
            abs(erp_to_power(56.99, 0., 0.) - 500_000.) / 500_000. < 1e-4
        )

if __name__ == "__main__":
    unittest.main()