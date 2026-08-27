import unittest

import numpy as np

from theia.util import angle_in_interval, erp_to_power


class UtilTest(unittest.TestCase):
    def test_erp_to_power(self):
        self.assertEqual(
            erp_to_power(30 - 2.15, 0.0, 0.0),
            1000.0,
        )
        self.assertTrue(
            abs(erp_to_power(56.99 - 2.15, 0.0, 0.0) - 500_000.0) / 500_000.0 < 1e-4
        )


class TestAngleInIntervalNormal(unittest.TestCase):
    """Intervals where low < high, no wraparound involved."""

    def test_inside_interval(self):
        self.assertTrue(angle_in_interval(45, -90, 90))

    def test_outside_interval(self):
        self.assertFalse(angle_in_interval(-100, -90, 90))

    def test_at_lower_bound(self):
        self.assertTrue(angle_in_interval(-90, -90, 90))

    def test_at_upper_bound(self):
        self.assertTrue(angle_in_interval(90, -90, 90))

    def test_just_outside_upper_bound(self):
        self.assertFalse(angle_in_interval(91, -90, 90))

    def test_just_outside_lower_bound(self):
        self.assertFalse(angle_in_interval(-91, -90, 90))


class TestAngleInIntervalWraparound(unittest.TestCase):
    """Intervals where low > high, e.g. [270, 90], wrapping through 0/360."""

    def test_inside_via_zero(self):
        self.assertTrue(angle_in_interval(0, 270, 90))

    def test_inside_above_zero(self):
        self.assertTrue(angle_in_interval(45, 270, 90))

    def test_inside_below_360(self):
        self.assertTrue(angle_in_interval(315, 270, 90))

    def test_outside_wraparound(self):
        self.assertFalse(angle_in_interval(180, 270, 90))

    def test_at_lower_bound_wraparound(self):
        self.assertTrue(angle_in_interval(270, 270, 90))

    def test_at_upper_bound_wraparound(self):
        self.assertTrue(angle_in_interval(90, 270, 90))

    def test_just_outside_wraparound(self):
        self.assertFalse(angle_in_interval(91, 270, 90))
        self.assertFalse(angle_in_interval(269, 270, 90))


class TestAngleInIntervalAnglesOutsideBaseRange(unittest.TestCase):
    """Angles given outside [0, 360) or negative, larger than period, etc."""

    def test_angle_greater_than_360(self):
        self.assertTrue(angle_in_interval(360 + 45, -90, 90))

    def test_angle_negative_wraps_correctly(self):
        self.assertTrue(angle_in_interval(-315, -90, 90))  # -315 == 45 mod 360

    def test_low_and_high_outside_base_range(self):
        # Equivalent to [-90, 90] but shifted by full turns
        self.assertTrue(angle_in_interval(45, -90 - 360, 90 + 360))


class TestAngleInIntervalDegenerateCases(unittest.TestCase):
    """Zero-width and full-circle intervals."""

    def test_zero_width_interval_matches_only_that_angle(self):
        self.assertTrue(angle_in_interval(30, 30, 30))
        self.assertFalse(angle_in_interval(31, 30, 30))
        self.assertFalse(angle_in_interval(29, 30, 30))

    def test_full_circle_interval_contains_everything(self):
        # A full circle expressed as e.g. [0, 360] should contain all angles.
        for a in (0, 45, 179, 270, 359, -30, 720):
            self.assertTrue(angle_in_interval(a, 0, 360))

    def test_full_circle_interval_other_representations(self):
        # Any interval whose span is a nonzero multiple of the period is
        # a full circle, regardless of where it starts.
        self.assertTrue(angle_in_interval(123, 90, 450))  # span = 360
        self.assertTrue(angle_in_interval(0, 360, 0))  # span = -360
        self.assertTrue(angle_in_interval(200, -180, 180))  # span = 360

    def test_full_circle_interval_array_input(self):
        angles = np.array([0, 90, 180, 270, 359])
        result = angle_in_interval(angles, 0, 360)
        self.assertTrue(np.all(result))
        self.assertEqual(result.dtype, np.bool_)
        self.assertEqual(result.shape, angles.shape)

    def test_zero_width_interval_is_not_treated_as_full_circle(self):
        # low == high exactly (span == 0) must still mean "just this one
        # angle", not "everywhere". This is what distinguishes it from
        # the [0, 360] full-circle case above.
        self.assertTrue(angle_in_interval(30, 30, 30))
        self.assertFalse(angle_in_interval(31, 30, 30))
        self.assertFalse(angle_in_interval(0, 30, 30))


class TestAngleInIntervalRadians(unittest.TestCase):
    """Same checks but using radians and period=2*pi."""

    def test_inside_interval_radians(self):
        self.assertTrue(angle_in_interval(0.1, -np.pi / 2, np.pi / 2, is_rad=True))

    def test_wraparound_radians(self):
        self.assertTrue(angle_in_interval(0, 3 * np.pi / 2, np.pi / 2, is_rad=True))

    def test_outside_wraparound_radians(self):
        self.assertFalse(
            angle_in_interval(np.pi, 3 * np.pi / 2, np.pi / 2, is_rad=True)
        )

    def test_full_circle_radians(self):
        self.assertTrue(angle_in_interval(1.23, 0, 2 * np.pi, is_rad=True))


class TestAngleInIntervalArrayInput(unittest.TestCase):
    """`angle` given as a numpy array should broadcast elementwise."""

    def test_array_normal_interval(self):
        angles = np.array([-100, -45, 0, 45, 91])
        expected = np.array([False, True, True, True, False])
        result = angle_in_interval(angles, -90, 90)
        np.testing.assert_array_equal(result, expected)

    def test_array_wraparound_interval(self):
        angles = np.array([0, 45, 180, 269, 270, 315, 90, 91])
        expected = np.array([True, True, False, False, True, True, True, False])
        result = angle_in_interval(angles, 270, 90)
        np.testing.assert_array_equal(result, expected)

    def test_array_result_dtype_is_bool(self):
        angles = np.array([0, 45, 180])
        result = angle_in_interval(angles, -90, 90)
        self.assertEqual(result.dtype, np.bool_)


if __name__ == "__main__":
    unittest.main()
