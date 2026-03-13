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
from theia.types import AbstractTracker


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

    def add_detections(self, detections: set[Detection]):
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


class DummyTracker(AbstractTracker):
    def add_detections(self, detections: list[Detection]):
        pass

    def get_tracks(self) -> list[theia.types.Track]:
        return []
