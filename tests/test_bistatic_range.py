import unittest

from matplotlib import pyplot as plt
import numpy as np

from theia.coordinates import LATLON_BOUNDS, CoordinateTransformations
from theia.data_loading_testing import load_pcl_reference_data
from theia.distance import get_bistatic_range
from theia.ellipsoid import Ellipsoid
from theia.terrain import SrtmTerrainModel


class BistaticRangeTest(unittest.TestCase):
    def test_bistatic_range(self):
        n_examples = 100
        rng = np.random.Generator(np.random.PCG64(463799))
        terrain_model = SrtmTerrainModel()

        # Sample within the boundaries of Switzerland (roughly).
        LAT_MIN, LAT_MAX = LATLON_BOUNDS["CH"]["lat"]
        LON_MIN, LON_MAX = LATLON_BOUNDS["CH"]["lon"]

        n = 0
        while n < n_examples:
            # Sample transmitter and emitter locations.
            p1 = terrain_model.sample_location(rng, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            p2 = terrain_model.sample_location(rng, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

            # Sample target position with flight height in [1., 10'000] m.
            target_position = terrain_model.sample_location(
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
            p_not_on_ellipsoid = terrain_model.sample_location(
                rng,
                LAT_MIN,
                LAT_MAX,
                LON_MIN,
                LON_MAX,
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

    def test_bistatic_range_against_reference(self):
        trajectories, detections = load_pcl_reference_data()
        for i, (detection, snr) in enumerate(detections):
            bistatic_range_ref = detection.bistatic_range
            bistatic_range_calculated, _, _, _ = get_bistatic_range(
                detection.sensor.transmitter.point.as_tuple(),
                detection.sensor.receiver.point.as_tuple(),
                detection.target.point.as_tuple(),
            )
            bistatic_range_calculated *= 1000.0  # [m]
            self.assertAlmostEqual(
                bistatic_range_calculated, bistatic_range_ref, delta=20
            )

    def test_plot_comparison(self):
        trajectories, detections_and_snr = load_pcl_reference_data()
        detections = [i[0] for i in detections_and_snr]
        assert len(trajectories) == 1
        trajectory = trajectories[0]

        txs = [d.sensor.transmitter for d in detections]
        rxs = [d.sensor.receiver for d in detections]

        txs_no_duplicates = []
        for tx in txs:
            if tx.id not in [t.id for t in txs_no_duplicates]:
                txs_no_duplicates.append(tx)
        rxs_no_duplicates = []
        for rx in rxs:
            if rx.id not in [r.id for r in rxs_no_duplicates]:
                rxs_no_duplicates.append(rx)

        for Tx in txs_no_duplicates:
            for Rx in rxs_no_duplicates:
                detections_ref_tx_rx = [
                    d
                    for d in detections
                    if d.sensor.transmitter.id == Tx.id
                    and d.sensor.receiver.id == Rx.id
                ]
                detection_times_ref = [
                    detection.time for detection in detections_ref_tx_rx
                ]
                bistatic_ranges_ref = [
                    detection.bistatic_range for detection in detections_ref_tx_rx
                ]
                bistatic_ranges_ref = np.asarray(bistatic_ranges_ref)

                bistatic_ranges = []
                target_pos_interpolated = []
                for t in detection_times_ref:
                    target = trajectory(t)
                    target_pos_interpolated.append(target.point)
                    r = (
                        get_bistatic_range(
                            Tx.point.as_tuple(),
                            Rx.point.as_tuple(),
                            target.point.as_tuple(),
                        )[0]
                        * 1000
                    )
                    bistatic_ranges.append(float(r))
                bistatic_ranges = np.asarray(bistatic_ranges)

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2 * 8, 4.5))

                ax = axes[0]
                ax.plot(detection_times_ref, bistatic_ranges_ref, label="openBURST")
                ax.plot(
                    detection_times_ref,
                    bistatic_ranges,
                    "-.",
                    label="mission execution",
                )
                ax.set_ylabel("Bistatic Range [m]", fontsize=16)

                ax = axes[1]
                ax.plot(
                    detection_times_ref,
                    bistatic_ranges_ref - bistatic_ranges,
                    label="openBURST - mission execution",
                )
                ax.set_ylabel("Difference in Bistatic Range [m]", fontsize=16)

                for ax in axes:
                    ax.set_xlabel("Time", fontsize=16)
                    ax.tick_params(axis="x", rotation=30)
                    ax.grid(True)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.legend(framealpha=1)

                fig.suptitle(f"Detections for Tx={Tx.id}, Rx={Rx.id}")
                fig.savefig(f"bistatic_range_Tx{Tx.id}_Rx{Rx.id}.png")


if __name__ == "__main__":
    unittest.main()
