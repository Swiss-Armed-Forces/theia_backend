import itertools
from pathlib import Path
import unittest

from theia.coordinates import CoordinateTransformations
from theia.data_loading import load_trajectory_file
from theia.measurement import MonostaticMeasurementTransformations
from theia.simulation.tracking import MonostaticPseudoTracker
from theia.test_data import get_uetliberg_radar
from theia.types import MonostaticRadarDetection

DELTA_LATLON = 1e-10
DELTA_ALT = 1e-5


class MonostaticPseudoTrackerTest(unittest.TestCase):
    def test_single_target_without_uncertainty(self):
        radar = get_uetliberg_radar()
        trajectories, _ = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/../data/data_opensky_2022-06-27.csv"
        )
        traj = next(traj for traj in trajectories if traj.target_id == 29)

        tracker = MonostaticPseudoTracker(removal_patience=1_000_000)
        for i, t in enumerate(traj.times):
            target = traj(t)
            p = CoordinateTransformations.geodetic_to_cartesian(
                target.lat,
                target.lon,
                target.alt,
            )
            elevation, azimuth, range_m = (
                MonostaticMeasurementTransformations.cartesian_to_elevation_azimuth_range(
                    radar.receiver.point,
                    p,
                )
            )
            detection = MonostaticRadarDetection(
                detection_id=i,
                time=t,
                radar=radar,
                target=target,
                snr=100.0,
                target_range=range_m,
                elevation_angle=elevation,
                azimuth_angle=azimuth,
                sigma_target_range=0.0,
                sigma_elevation=0.0,
                sigma_azimuth=0.0,
            )
            tracker.add_detections([detection])

            if i == 0:
                # A single detection cannot be a track due to interpolation.
                tracks = tracker.get_tracks()
                self.assertEqual(len(tracks), 0)
            else:
                tracks = tracker.get_tracks()
                self.assertEqual(len(tracks), 1)
                track = tracks[0]
                self.assertEqual(track.id, str(target.id))
                self.assertEqual(len(track.states), i + 1)

        for t in traj.times:
            target = traj(t)
            lat_true, lon_true, alt_true = target.point.as_tuple()
            x, vx, y, vy, z, vz = track(t)
            lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
            self.assertAlmostEqual(lat_true, lat, delta=DELTA_LATLON)
            self.assertAlmostEqual(lon_true, lon, delta=DELTA_LATLON)
            self.assertAlmostEqual(alt_true, alt, delta=DELTA_ALT)

    def test_multiple_targets_without_uncertainty(self):
        radar = get_uetliberg_radar()
        trajectories, _ = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/../data/data_opensky_2022-06-27.csv"
        )
        trajectories = sorted(trajectories, key=lambda t: t.target_id)
        times = set(itertools.chain.from_iterable([t.times for t in trajectories]))
        times = sorted(list(times))

        tracker = MonostaticPseudoTracker(removal_patience=1_000_000)
        for i, t in enumerate(times):
            detections = []
            for trajectory in trajectories:
                if t < trajectory.times[0] or t > trajectory.times[-1]:
                    continue
                target = trajectory(t)
                p = CoordinateTransformations.geodetic_to_cartesian(
                    target.lat,
                    target.lon,
                    target.alt,
                )
                elevation, azimuth, range_m = (
                    MonostaticMeasurementTransformations.cartesian_to_elevation_azimuth_range(
                        radar.receiver.point,
                        p,
                    )
                )
                detection = MonostaticRadarDetection(
                    detection_id=i,
                    time=t,
                    radar=radar,
                    target=target,
                    snr=100.0,
                    target_range=range_m,
                    elevation_angle=elevation,
                    azimuth_angle=azimuth,
                    sigma_target_range=0.0,
                    sigma_elevation=0.0,
                    sigma_azimuth=0.0,
                )
                detections.append(detection)
            tracker.add_detections(detections)

        tracks = tracker.get_tracks()

        for trajectory in trajectories:
            track = next(
                track for track in tracks if int(track.id) == trajectory.target_id
            )
            for t in trajectory.times:
                target = trajectory(t)
                lat_true, lon_true, alt_true = target.point.as_tuple()
                x, vx, y, vy, z, vz = track(t)
                lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
                self.assertAlmostEqual(lat_true, lat, delta=DELTA_LATLON)
                self.assertAlmostEqual(lon_true, lon, delta=DELTA_LATLON)
                self.assertAlmostEqual(alt_true, alt, delta=DELTA_ALT)


if __name__ == "__main__":
    unittest.main()
