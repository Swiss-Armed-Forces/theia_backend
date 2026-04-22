import datetime
import itertools

import numpy as np
import stonesoup
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater, LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

import theia
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.measurement import MonostaticMeasurementTransformations
from theia.types import (
    CLUTTER_TARGET,
    AbstractTracker,
    MonostaticRadarDetection,
    PclDetection,
    Point,
    Track,
)


class SingleTargetEcefTracker:
    """
    A simple Kalman tracker for a single target in ECEF space.

    No association takes place.
    """

    def __init__(
        self,
        id: int,
        t0: datetime.datetime,
        pos0: tuple[float, float, float],
        v_max: float = 300,
    ):
        """
        Parameters
        ----------
        id: int
            Target ID
        t0: datetime.datetime
            Time at which the prior is assumed
        pos0: tuple[float, float, float]
            Prior for the position at t0
        v_max: float, default 300
            Maximum expected target speed [m / s]
        """
        self._id = id
        q = 1.0
        transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(q), ConstantVelocity(q), ConstantVelocity(q)]
        )
        self._predictor = ExtendedKalmanPredictor(transition_model)
        self._updater = ExtendedKalmanUpdater()

        # Define initialiser:
        # The initial velocity uncertaintyis a worst-case estimate:
        # The standard deviation is chosen such that 3-sigma encompasses
        # the entire expected velocity range (99.7% confidence interval).
        # Generate new track from unassigned detections if at least
        # five detections are assigned to it.
        sigma_v = v_max / 3

        # Used by default, but measured components are replaced by
        # the detection that initialises the track.
        self._prior = GaussianState(
            [[pos0[0]], [0], [pos0[1]], [0], [pos0[2]], [0]],
            np.diag([0.0, sigma_v**2, 0.0, sigma_v**2, 0.0, sigma_v**2]),
            timestamp=t0,
        )

        self._track = stonesoup.types.track.Track()

    def add_detection(
        self,
        detection: Detection,
    ):
        prediction = self._predictor.predict(self._prior, timestamp=detection.timestamp)
        hypothesis = SingleHypothesis(prediction, detection)
        posterior = self._updater.update(hypothesis)
        self._track.append(posterior)
        self._prior = self._track[-1]

    def to_theia_track(self) -> Track | None:
        states = [(s.timestamp, s.state_vector.flatten()) for s in self._track.states]
        if len(states) < 2:
            return None
        else:
            return theia.types.Track(id=str(self._id), states=states)


class PseudoTracker(AbstractTracker):
    r"""
    A pseudo-tracker for quick iteration.

    This class ignores clutter and assumes that data association is perfect.
    Tracking is executed in Cartesian ECEF space using a Kalman filter.

    Notes
    -----
    See :ref:`pseudo_tracker` for documentation of the algorithm.
    """

    def __init__(
        self,
        removal_patience: int,
        rng: np.random.Generator,
        start_time: datetime.datetime,
        prior_position: Point = Point(
            lat=POSITIONS_OF_INTEREST["CH_CENTER"]["lat"],
            lon=POSITIONS_OF_INTEREST["CH_CENTER"]["lon"],
            alt=1000.0,
        ),
    ):
        """
        Parameters
        ----------
        removal_patience: int
            How many iterations without update a track survives
        rng: np.random.Generator
            Random number generator
        start_time: datetime.datetime
            Start time of the simulation (time of the prior)
        prior_position: Point, default Point(
                lat=POSITIONS_OF_INTEREST["CH_CENTER"]["lat"],
                lon=POSITIONS_OF_INTEREST["CH_CENTER"]["lon"],
                alt=1000.0,
            )
            Position to serve as prior
        """
        self._trackers: dict[int, SingleTargetEcefTracker] = {}
        """Tracker per target ID"""
        self._iterations_without_update: dict[int, int] = {}
        """How many iterations (value) since the last update per target ID (key)"""
        self._removal_patience = removal_patience
        """Number of iterations a track is kept without updates"""
        self._rng = rng
        self._t0 = start_time
        self._prior = CoordinateTransformations.geodetic_to_cartesian(
            *prior_position.as_tuple()
        )

    def _monostatic_init_update(
        self,
        detections: list[MonostaticRadarDetection],
    ) -> Detection:
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
        measured_state = np.array([x, y, z])

        sigma = np.max(
            (
                detection.sigma_target_range,
                detection.sigma_azimuth * detection.target_range,
                detection.sigma_elevation * detection.target_range,
            )
        )
        covar = np.diag([sigma, sigma, sigma])

        return Detection(
            measured_state,
            measurement_model=LinearGaussian(
                ndim_state=6, mapping=(0, 2, 4), noise_covar=covar
            ),
            timestamp=detection_time,
        )

    def _pcl_init_update(self, detections: list[PclDetection]) -> Detection:
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
        measured_state = np.array([x, y, z])

        covar = np.diag([sigma, sigma, sigma])

        return Detection(
            measured_state,
            measurement_model=LinearGaussian(
                ndim_state=6, mapping=(0, 2, 4), noise_covar=covar
            ),
            timestamp=detection_time,
        )

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
                detection = self._monostatic_init_update(
                    grouped_monostatic_detections[target_id]
                )
            elif target_id in grouped_pcl_detections:
                detections = grouped_pcl_detections[target_id]
                detections = sorted(detections, key=lambda d: d.bistatic_range)
                if target_id in self._trackers or len(detections) >= 3:
                    detection = self._pcl_init_update(detections)
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
            tracker = self._trackers.setdefault(
                target_id, SingleTargetEcefTracker(target_id, self._t0, self._prior)
            )
            tracker.add_detection(detection)
            self._iterations_without_update[target_id] = 0
            updated_targets.add(target_id)

        # Remove targets that haven't been updated in a while.
        # The list() is important: It allows to modify the states dict during iteration.
        for target_id in list(self._trackers.keys()):
            if target_id not in updated_targets:
                self._iterations_without_update[target_id] += 1
                if self._iterations_without_update[target_id] > self._removal_patience:
                    del self._trackers[target_id]
                    del self._iterations_without_update[target_id]

    def get_tracks(self) -> list[Track]:
        tracks = [tracker.to_theia_track() for tracker in self._trackers.values()]
        tracks = [track for track in tracks if track is not None]
        return tracks
