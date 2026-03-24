import datetime
import itertools

import numpy as np
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import Hypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

import theia
from theia.measurement import MonostaticMeasurementTransformations
from theia.stonesoup_interface import MonostaticDetectionFactory
from theia.types import CLUTTER_TARGET, AbstractTracker, MonostaticRadarDetection, Point


class DummyTracker(AbstractTracker):
    def add_detections(self, detections):
        pass

    def get_tracks(self):
        return []


class MonostaticSingleSensorTracker(AbstractTracker):
    def __init__(
        self,
        q: float = 1.0,
        mahalanobis_miss_distance: float = 50.0,
        deletion_covariance_threshold=200_000.0,
    ):
        # Initialise the tracker.
        transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(q), ConstantVelocity(q), ConstantVelocity(q)]
        )
        predictor = ExtendedKalmanPredictor(transition_model)
        self._updater = ExtendedKalmanUpdater()

        hypothesiser = DistanceHypothesiser(
            predictor,
            self._updater,
            measure=Mahalanobis(),
            missed_distance=mahalanobis_miss_distance,
        )
        self._data_associator = NearestNeighbour(hypothesiser)

        # Delete tracks once their covariance trace exceeds 4.
        self._deleter = CovarianceBasedDeleter(
            covar_trace_thresh=deletion_covariance_threshold
        )
        # Tentative tracks — kill immediately if no match on next scan.
        init_deleter = UpdateTimeStepsDeleter(time_steps_since_update=1)

        # Define initialiser:
        # The initial velocity uncertaintyis a worst-case estimate:
        # The standard deviation is chosen such that 3-sigma encompasses
        # the entire expected velocity range (99.7% confidence interval).
        # Generate new track from unassigned detections if at least
        # five detections are assigned to it.
        v_max = 300
        sigma_v = v_max / 3

        self._initiator = MultiMeasurementInitiator(
            # Used by default, but measured components are replaced by
            # the detection that initialises the track.
            prior_state=GaussianState(
                [[0], [0], [0], [0], [0], [0]],
                np.diag([0.0, sigma_v**2, 0.0, sigma_v**2, 0.0, sigma_v**2]),
            ),
            deleter=init_deleter,
            data_associator=self._data_associator,
            updater=self._updater,
            min_points=5,
        )

        # Initialise the tracks.
        self._tracks: set[Track] = set()

    def add_detections(self, detections: set[MonostaticRadarDetection]):
        detections = set([MonostaticDetectionFactory.from_theia(d) for d in detections])
        if len(detections) == 0:
            return
        any_detection = next(iter(detections))
        time = any_detection.timestamp
        assert all([d.timestamp == any_detection.timestamp for d in detections])
        hypotheses: dict[Track, Hypothesis] = self._data_associator.associate(
            self._tracks,
            detections,
            time,
        )
        associated_measurements: Detection = set()
        for track in self._tracks:
            hypothesis: Hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = self._updater.update(hypothesis)
                track.append(post)
                associated_measurements.add(hypothesis.measurement)
            else:
                # When data associator says no detections are good enough,
                # we'll keep the prediction.
                track.append(hypothesis.prediction)

        # Carry out deletion and initiation.
        self._tracks -= self._deleter.delete_tracks(self._tracks)

        unassociated = detections - associated_measurements

        if unassociated:
            new_tracks = self._initiator.initiate(unassociated, time)
            self._tracks |= new_tracks

    def get_tracks(self) -> list[theia.types.Track]:
        tracks: list[theia.types.Track] = []
        for track in self._tracks:
            states = [(s.timestamp, s.state_vector.flatten()) for s in track.states]
            tracks.append(theia.types.Track(id=track.id, states=states))
        return tracks


class MonostaticPseudoTracker(AbstractTracker):
    """
    A pseudo-tracker for quick iteration.

    This class ignores clutter and assumes that track init and data association
    are perfect. It simply keeps one track per target and adds a sample for each
    update (if any). If multiple radars detect the same target, the position with
    the most recent time stamp is used. If multiple detections share the same
    time stamp,the one with the least range is used (heuristic).
    """

    def __init__(self):
        self._states: dict[int, list[tuple[datetime.datetime, np.ndarray]]] = {}
        """History of positions per target ID"""

    def add_detections(self, detections):
        def f(d: MonostaticRadarDetection):
            return d.target.id

        detections = sorted(detections, key=f)
        for target_id, target_detections in itertools.groupby(detections, f):
            if target_id == CLUTTER_TARGET.id:
                continue
            detection = sorted(
                target_detections,
                # Minus sign: Needed because we sort descending in time.
                key=lambda d: (-d.time.timestamp(), d.target_range),
            )[0]
            x, y, z = (
                MonostaticMeasurementTransformations.elevation_azimuth_range_to_cartesian(
                    detection.radar.receiver.point,
                    detection.elevation_angle,
                    detection.azimuth_angle,
                    detection.target_range,
                )
            )
            # TODO: Set actual velocities instead of zero.
            state = np.array([x, 0.0, y, 0.0, z, 0.0])
            history = self._states.setdefault(target_id, [])
            history.append((detection.time, state))

    def get_tracks(self) -> list[theia.types.Track]:
        tracks: list[theia.types.Track] = []
        for target_id, timepoints in self._states.items():
            if len(timepoints) < 2:
                continue
            tracks.append(theia.types.Track(id=str(target_id), states=timepoints))
        return tracks
