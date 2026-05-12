import datetime
import itertools

import numpy as np
import pydantic
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
from theia.distance import line_of_sight_distance
from theia.types import (
    CLUTTER_TARGET,
    AbstractTracker,
    MonostaticRadarDetection,
    PclDetection,
    PetDetection,
    Point,
    Target,
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
        sidc: str = "",
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
        sidc: str, default ""
            Symbol identification code representing the target
        """
        self._id = id
        self._sidc = sidc
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
        self._has_detections = False

        self._track = stonesoup.types.track.Track()

    def add_detection(
        self,
        detection: Detection,
    ):
        if not self._has_detections:
            self._prior.state_vector[[0, 2, 4]] = detection.state_vector
            self._has_detections = True
        prediction = self._predictor.predict(self._prior, timestamp=detection.timestamp)
        hypothesis = SingleHypothesis(prediction, detection)
        posterior = self._updater.update(hypothesis)
        self._track.append(posterior)
        self._prior = self._track[-1]

    def to_theia_track(self) -> Track | None:
        states = [
            (s.timestamp, np.array(s.state_vector.flatten()))
            for s in self._track.states
        ]
        if len(states) <= 1:
            return None
        else:
            return theia.types.Track(id=str(self._id), sidc=self._sidc, states=states)


class TargetDetections(pydantic.BaseModel):
    target_id: int
    target_sidc: str
    monostatic_detections: list[MonostaticRadarDetection] = []
    pcl_detections: list[PclDetection] = []
    pet_detections: list[PetDetection] = []


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

    def _sample_position(
        self,
        target: Target,
        sigmas: tuple[float, float, float],
        detection_time: datetime.datetime,
    ) -> tuple[float, float, float]:
        mean = CoordinateTransformations.geodetic_to_cartesian(
            target.lat,
            target.lon,
            target.alt,
        )
        # Sample a fake detection.
        x, y, z = self._rng.normal(mean, sigmas)
        measured_state = np.array([x, y, z])

        covar = np.diag(sigmas)

        return Detection(
            measured_state,
            measurement_model=LinearGaussian(
                ndim_state=6, mapping=(0, 2, 4), noise_covar=covar
            ),
            timestamp=detection_time,
        )

    def _monostatic_detections_to_ecef(
        self,
        detections: list[MonostaticRadarDetection],
    ) -> Detection:
        """
        Summarize monostatic detections to a single detection in ECEF space.

        Parameters
        ----------
        detections: list[MonostaticRadarDetection]
            Detections for the same target. Must be of length > 0.

        Returns
        -------
        stonesoup.types.detection.Detection
            A detection representing a position in 3D space, including measurement model

        Raises
        ------
        ValueError
            If `len(detections) == 0`
        """
        if len(detections) == 0:
            raise ValueError(
                "Need at least 1 RAD detection to estimate target position in ECEF space"
            )
        # Select detection with shortest range and initiate or update
        # the track.
        detection = sorted(detections, key=lambda d: d.target_range)[0]

        sigma = np.max(
            (
                detection.sigma_target_range,
                detection.sigma_azimuth * detection.target_range,
                detection.sigma_elevation * detection.target_range,
            )
        )
        detection = detections[0]
        return self._sample_position(
            detection.target,
            (sigma, sigma, sigma),
            detection.time,
        )

    def _pcl_detections_to_ecef(self, detections: list[PclDetection]) -> Detection:
        """
        Sample a position in ECEF space around the ground truth one using
        error propagation.

        Parameters
        ----------
        detections: list[PclDetection]
            Detections for the same target. Must be of length >= 3.

        Returns
        -------
        stonesoup.types.detection.Detection
            A detection representing a position in 3D space, including measurement model

        Raises
        ------
        ValueError
            If `len(detections) < 3`
        """
        if len(detections) < 3:
            raise ValueError(
                "Need at least 3 PCL detections to estimate target position in ECEF space"
            )

        detections = sorted(detections, key=lambda d: d.bistatic_range)
        sigma = np.max([d.sigma_bistatic_range for d in detections])
        # All detections share the same target, i. e. we can select
        # any of them to access the true target state.
        detection = detections[0]
        return self._sample_position(
            detection.target,
            (sigma, sigma, sigma),
            detection.time,
        )

    def _pet_detections_to_ecef(self, detections: list[PetDetection]) -> Detection:
        """
        Sample a position in ECEF space around the ground truth one using
        error propagation.

        Parameters
        ----------
        detections: list[PetDetection]
            Detections for the same target. Must be of length >= 2.

        Returns
        -------
        stonesoup.types.detection.Detection
            A detection representing a position in 3D space, including measurement model

        Raises
        ------
        ValueError
            If `len(detections) < 2`
        """
        if len(detections) < 2:
            raise ValueError(
                "Need at least 2 PET detections to estimate target position in ECEF space"
            )

        sigmas = []
        for detection in detections:
            d = line_of_sight_distance(
                detection.target.lat,
                detection.target.lon,
                detection.target.alt,
                detection.radar.receiver.lat,
                detection.radar.receiver.lon,
                detection.radar.receiver.alt,
            )
            sigmas.append(d * detection.azimuth)
            sigmas.append(d * detection.elevation)
        sigma = np.max(sigmas)

        # All detections share the same target, i. e. we can select
        # any of them to access the true target state.
        detection = detections[0]
        return self._sample_position(
            detection.target,
            (sigma, sigma, sigma),
            detection.time,
        )

    def _combined_pcl_pet_detections_to_ecef(
        self,
        pcl_detections: tuple[PclDetection, PclDetection],
        pet_detection: PetDetection,
    ) -> Detection:
        """
        Sample a position in ECEF space around the ground truth one using
        error propagation.

        Parameters
        ----------
        pcl_detections: tuple[PclDetection, PclDetection]
            PCL detections
        pet_detection: PetDetection
            PET detection

        Returns
        -------
        stonesoup.types.detection.Detection
            A detection representing a position in 3D space, including measurement model

        Raises
        ------
        ValueError
            If `len(detections) < 2`
        """
        d = line_of_sight_distance(
            pet_detection.target.lat,
            pet_detection.target.lon,
            pet_detection.target.alt,
            pet_detection.radar.receiver.lat,
            pet_detection.radar.receiver.lon,
            pet_detection.radar.receiver.alt,
        )
        sigma = np.max(
            (
                d * pet_detection.azimuth,
                d * pet_detection.elevation,
                pcl_detections[0].sigma_bistatic_range,
                pcl_detections[1].sigma_bistatic_range,
            )
        )

        # All detections share the same target, i. e. we can select
        # any of them to access the true target state.
        detection = pet_detection
        return self._sample_position(
            detection.target,
            (sigma, sigma, sigma),
            detection.time,
        )

    def _preprocess_detections(
        self,
        monostatic_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
    ) -> list[TargetDetections]:
        # Ignore clutter.
        monostatic_detections = [
            d for d in monostatic_detections if d.target != CLUTTER_TARGET
        ]
        pcl_detections = [d for d in pcl_detections if d.target != CLUTTER_TARGET]
        pet_detections = [d for d in pet_detections if d.target != CLUTTER_TARGET]

        def f(d: MonostaticRadarDetection | PclDetection | PetDetection) -> int:
            return d.target.id

        monostatic_detections = sorted(monostatic_detections, key=f)
        pcl_detections = sorted(pcl_detections, key=f)
        pet_detections = sorted(pet_detections, key=f)

        grouped_monostatic_detections = {
            key: list(values)
            for key, values in itertools.groupby(monostatic_detections, f)
        }
        grouped_pcl_detections = {
            key: list(values) for key, values in itertools.groupby(pcl_detections, f)
        }
        grouped_pet_detections = {
            key: list(values) for key, values in itertools.groupby(pet_detections, f)
        }

        monostatic_target_ids = set(grouped_monostatic_detections.keys())
        pcl_target_ids = set(grouped_pcl_detections)
        pet_target_ids = set(grouped_pet_detections)

        target_ids = list(
            monostatic_target_ids.union(pcl_target_ids).union(pet_target_ids)
        )

        target_detections: list[TargetDetections] = []
        for target_id in target_ids:
            if len(grouped_monostatic_detections.get(target_id, [])) > 0:
                target = grouped_monostatic_detections[target_id][0].target
            elif len(grouped_pcl_detections.get(target_id, [])) > 0:
                target = grouped_pcl_detections[target_id][0].target
            elif len(grouped_pet_detections.get(target_id, [])) > 0:
                target = grouped_pet_detections[target_id][0].target
            else:
                raise RuntimeError("This code should never be reached!")

            target_detections.append(
                TargetDetections(
                    target_id=target_id,
                    target_sidc=target.sidc,
                    monostatic_detections=grouped_monostatic_detections.get(
                        target_id, []
                    ),
                    pcl_detections=grouped_pcl_detections.get(target_id, []),
                    pet_detections=grouped_pet_detections.get(target_id, []),
                )
            )
        return target_detections

    def add_detections(
        self,
        monostatic_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
    ):
        target_detections = self._preprocess_detections(
            monostatic_detections,
            pcl_detections,
            pet_detections,
        )
        updated_targets: set[int] = set()
        for detections in target_detections:
            track_exists = detections.target_id in self._trackers
            detection: Detection | None = None
            if len(detections.monostatic_detections) > 0:
                # Whenever we have monostatic detections, we use those detections
                # to initialize or update the track.
                detection = self._monostatic_detections_to_ecef(
                    detections.monostatic_detections
                )
            elif len(detections.pcl_detections) >= 3:
                detection = self._pcl_detections_to_ecef(detections.pcl_detections)
            elif len(detections.pet_detections) >= 2:
                detection = self._pet_detections_to_ecef(detections.pet_detections)
            elif (
                len(detections.pcl_detections) == 2
                and len(detections.pet_detections) == 1
            ):
                detection = self._combined_pcl_pet_detections_to_ecef(
                    tuple(detections.pcl_detections),
                    detections.pet_detections[0],
                )
            elif (len(detections.pcl_detections) == 2) and track_exists:
                detection = self._

            # Actually initialise or update the track.
            if detection is not None:
                tracker = self._trackers.setdefault(
                    detections.target_id,
                    SingleTargetEcefTracker(
                        detections.target_id,
                        self._t0,
                        self._prior,
                        sidc=detections.target_sidc,
                    ),
                )
                tracker.add_detection(detection)
                self._iterations_without_update[detections.target_id] = 0
                updated_targets.add(detections.target_id)

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
