import unittest

import numpy as np

from theia.coordinates import LATLON_BOUNDS, CoordinateTransformations, sample_location
from theia.distance import get_bistatic_range
from theia.ellipsoid import Ellipsoid
from theia.types import Point, Target


class BistaticRangeTest(unittest.TestCase):
    def test_bistatic_range(self):
        n_examples = 100
        rng = np.random.Generator(np.random.PCG64(463799))

        # Sample within the boundaries of Switzerland (roughly).
        LAT_MIN, LAT_MAX = LATLON_BOUNDS["CH"]["lat"]
        LON_MIN, LON_MAX = LATLON_BOUNDS["CH"]["lon"]

        n = 0
        while n < n_examples:
            # Sample transmitter and emitter locations.
            p1 = sample_location(rng, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            p2 = sample_location(rng, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

            # Sample target position with flight height in [1., 10'000] m.
            target_position = sample_location(
                rng,
                LAT_MIN,
                LAT_MAX,
                LON_MIN,
                LON_MAX,
            )
            meters_above_ground_level = rng.uniform(1.0, 10_000)
            target_position.alt += meters_above_ground_level

            p1_xyz = CoordinateTransformations.geodetic_to_cartesian(*p1.as_tuple())
            p2_xyz = CoordinateTransformations.geodetic_to_cartesian(*p2.as_tuple())
            target_position_xyz = CoordinateTransformations.geodetic_to_cartesian(
                *target_position.as_tuple()
            )

            # Calculate bistatic range and make sure that the target lies on the
            # corresponding ellipsoid.
            bistatic_range, d_rx_tgt, d_tx_tgt, d_rx_tx = get_bistatic_range(
                p1.as_tuple(),
                p2.as_tuple(),
                target_position.as_tuple(),
            )

            actual_bistatic_range = (d_rx_tgt + d_tx_tgt) * 1000

            e = Ellipsoid(p1_xyz, p2_xyz, actual_bistatic_range)

            self.assertTrue(
                e.is_on_surface(
                    target_position_xyz,
                    point_in_world_coord=True,
                ),
            )

            # Sanity check to ensure that the on-ellipse-surface test is actually
            # correct.
            p_not_on_ellipsoid = sample_location(
                rng, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
            )
            p_not_on_ellipsoid = CoordinateTransformations.geodetic_to_cartesian(
                *p_not_on_ellipsoid.as_tuple()
            )
            self.assertFalse(
                e.is_on_surface(
                    p_not_on_ellipsoid,
                    point_in_world_coord=True,
                ),
            )

            n += 1


if __name__ == "__main__":
    unittest.main()
