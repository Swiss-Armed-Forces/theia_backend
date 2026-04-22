import datetime
import itertools

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.measurement import MonostaticMeasurementTransformations
from theia.types import (
    CLUTTER_TARGET,
    AbstractTracker,
    MonostaticRadarDetection,
    PclDetection,
    Track,
)


class PseudoTracker(AbstractTracker):
    r"""
    A pseudo-tracker for quick iteration.

    This class ignores clutter and assumes that data association is perfect.
    Tracking is executed in Cartesian ECEF space using a Kalman filter.

    Notes
    -----
    See :ref:`pseudo_tracker` for documentation of the algorithm.
    """

    def __init__(self, removal_patience: int, rng: np.random.Generator):
        """
        Parameters
        ----------
        removal_patience: int
            How many iterations without update a track survives
        rng: np.random.Generator
            Random number generator
        """
        self._states: dict[int, list[tuple[datetime.datetime, np.ndarray]]] = {}
        """History of Cartesian ECEF positions per target ID"""
        self._iterations_without_update: dict[int, int] = {}
        """How many iterations (value) since the last update per target ID (key)"""
        self._removal_patience = removal_patience
        """Number of iterations a track is kept without updates"""
        self._rng = rng

    def _monostatic_init_update(
        self,
        detections: list[MonostaticRadarDetection],
    ) -> tuple[datetime.datetime, np.ndarray]:
        # Select detection with shortest range and initiate or update
        # the track.
        detection = sorted(detections, key=lambda d: d.target_range)[0]
        detection_time = detection.time

        x, y, z = (
            MonostaticMeasurementTransformations.elevation_azimuth_range_to_cartesian(
                detection.radar.receiver.point,
                detection.elevation_angle,
                detection.azimuth_angle,
                detection.target_range,
            )
        )
        # TODO: Set actual velocities instead of zero.
        # x, vx, y, vy, z, vz (order as in stonesoup)
        state = np.array([x, 0.0, y, 0.0, z, 0.0])

        return detection_time, state

    def _pcl_init_update(
        self, detections: list[PclDetection]
    ) -> tuple[datetime.datetime, np.ndarray]:
        # We sample a position around the real one using error propagation.
        # All detections share the same target, i. e. we can select
        # any of them to access the true target state.
        # Finally, a fake detection is created.
        sigma = np.max([d.sigma_bistatic_range for d in detections])
        detection = detections[0]
        detection_time = detection.time
        target = detection.target
        mean = CoordinateTransformations.geodetic_to_cartesian(
            target.lat,
            target.lon,
            target.alt,
        )
        x, y, z = self._rng.normal(mean, [sigma, sigma, sigma])
        # TODO: Set actual velocities instead of zero.
        # x, vx, y, vy, z, vz (order as in stonesoup)
        state = np.array([x, 0.0, y, 0.0, z, 0.0])

        return detection_time, state

    def add_detections(
        self,
        monostatic_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
    ):
        # Ignore clutter.
        monostatic_detections = [
            d for d in monostatic_detections if d.target != CLUTTER_TARGET
        ]
        pcl_detections = [d for d in pcl_detections if d.target != CLUTTER_TARGET]

        def f(d: MonostaticRadarDetection | PclDetection) -> int:
            return d.target.id

        monostatic_detections = sorted(monostatic_detections, key=f)
        pcl_detections = sorted(pcl_detections, key=f)

        grouped_monostatic_detections = {
            key: list(values)
            for key, values in itertools.groupby(monostatic_detections, f)
        }
        grouped_pcl_detections = {
            key: list(values) for key, values in itertools.groupby(pcl_detections, f)
        }

        monostatic_target_ids = set(grouped_monostatic_detections.keys())
        pcl_target_ids = set(grouped_pcl_detections)

        target_ids = list(monostatic_target_ids.union(pcl_target_ids))

        updated_targets: set[int] = set()
        for target_id in target_ids:
            if target_id in grouped_monostatic_detections:
                detection_time, state = self._monostatic_init_update(
                    grouped_monostatic_detections[target_id]
                )
            elif target_id in grouped_pcl_detections:
                detections = grouped_pcl_detections[target_id]
                detections = sorted(detections, key=lambda d: d.bistatic_range)
                if target_id in self._states or len(detections) >= 3:
                    detection_time, state = self._pcl_init_update(detections)
                else:
                    # We do not have a track yet, but there are not enough
                    # detections to initialize a new one.
                    continue
            else:
                raise RuntimeError(
                    "Unexpected behaviour: This code should never be reached!"
                )

            # Actually initialise or update the track.
            # TODO: Use a Kalman filter.
            history = self._states.setdefault(target_id, [])
            history.append((detection_time, state))
            self._iterations_without_update[target_id] = 0
            updated_targets.add(target_id)

        # Remove targets that haven't been updated in a while.
        # The list() is important: It allows to modify the states dict during iteration.
        for target_id in list(self._states.keys()):
            if target_id not in updated_targets:
                self._iterations_without_update[target_id] += 1
                if self._iterations_without_update[target_id] > self._removal_patience:
                    del self._states[target_id]
                    del self._iterations_without_update[target_id]

    def get_tracks(self) -> list[Track]:
        tracks: list[Track] = []
        for target_id, timepoints in self._states.items():
            # print([tp[1].tolist() for tp in timepoints])
            if len(timepoints) < 2:
                continue
            tracks.append(Track(id=str(target_id), states=timepoints))
        return tracks
