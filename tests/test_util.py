import unittest

from theia.util import erp_to_power


class UtilTest(unittest.TestCase):
    def test_erp_to_power(self):
        self.assertEqual(
            erp_to_power(30-2.15, 0.0, 0.0),
            1000.0,
        )
        self.assertTrue(
            abs(erp_to_power(56.99-2.15, 0.0, 0.0) - 500_000.0) / 500_000.0 < 1e-4
        )


if __name__ == "__main__":
    unittest.main()
