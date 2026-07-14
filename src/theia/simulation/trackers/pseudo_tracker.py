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
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

import theia
from theia.coordinates import (
    POSITIONS_OF_INTEREST,
    CoordinateTransformations,
)
from theia.detection.ecef_sampler import EcefDetectionSampler
from theia.types import (
    CLUTTER_TARGET,
    AbstractTracker,
    Entity,
    IdProvider,
    MonostaticRadarDetection,
    PclDetection,
    PetDetection,
    Point,
    TrackInitEvent,
    Trigger,
    VisualDetection,
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
    visual_detections: list[VisualDetection] = []


class PseudoTracker(AbstractTracker, Trigger):
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
        super().__init__()
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
        self._sampler = EcefDetectionSampler(rng)

    def _preprocess_detections(
        self,
        monostatic_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        visual_detections: list[VisualDetection],
    ) -> list[TargetDetections]:
        # Ignore clutter.
        monostatic_detections = [
            d for d in monostatic_detections if d.target != CLUTTER_TARGET
        ]
        pcl_detections = [d for d in pcl_detections if d.target != CLUTTER_TARGET]
        pet_detections = [d for d in pet_detections if d.target != CLUTTER_TARGET]

        def f(
            d: MonostaticRadarDetection | PclDetection | PetDetection | VisualDetection,
        ) -> int:
            return d.target.id

        monostatic_detections = sorted(monostatic_detections, key=f)
        pcl_detections = sorted(pcl_detections, key=f)
        pet_detections = sorted(pet_detections, key=f)
        visual_detections = sorted(visual_detections, key=f)

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
        grouped_visual_detections = {
            key: list(values) for key, values in itertools.groupby(visual_detections, f)
        }

        monostatic_target_ids = set(grouped_monostatic_detections.keys())
        pcl_target_ids = set(grouped_pcl_detections)
        pet_target_ids = set(grouped_pet_detections)
        visual_target_ids = set(grouped_visual_detections)

        target_ids = list(
            monostatic_target_ids.union(pcl_target_ids)
            .union(pet_target_ids)
            .union(visual_target_ids)
        )

        target_detections: list[TargetDetections] = []
        for target_id in target_ids:
            if len(grouped_monostatic_detections.get(target_id, [])) > 0:
                target = grouped_monostatic_detections[target_id][0].target
            elif len(grouped_pcl_detections.get(target_id, [])) > 0:
                target = grouped_pcl_detections[target_id][0].target
            elif len(grouped_pet_detections.get(target_id, [])) > 0:
                target = grouped_pet_detections[target_id][0].target
            elif len(grouped_visual_detections.get(target_id, [])) > 0:
                target = grouped_visual_detections[target_id][0].target
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
                    visual_detections=grouped_visual_detections.get(target_id, []),
                )
            )
        return target_detections

    def add_detections(
        self,
        monostatic_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        visual_detections: list[VisualDetection],
        id_provider: IdProvider,  # needed to get unused track IDs
    ):
        target_detections = self._preprocess_detections(
            monostatic_detections,
            pcl_detections,
            pet_detections,
            visual_detections,
        )
        updated_targets: set[int] = set()
        for detections in target_detections:
            track_exists = detections.target_id in self._trackers
            detection = self._sampler.sample(
                detections.monostatic_detections,
                detections.pcl_detections,
                detections.pet_detections,
                detections.visual_detections,
                track_exists,
            )

            # Actually initialise or update the track.
            if detection is not None:
                track_id = id_provider.increment(Entity.TRACK)
                tracker = self._trackers.setdefault(
                    detections.target_id,
                    SingleTargetEcefTracker(
                        track_id,
                        self._t0,
                        self._prior,
                        sidc=detections.target_sidc,
                    ),
                )
                tracker.add_detection(detection)
                self._iterations_without_update[detections.target_id] = 0
                updated_targets.add(detections.target_id)

                if len(tracker._track.states) == 1:
                    self._broadcast_event(
                        TrackInitEvent(
                            id=id_provider.increment(Entity.EVENT),
                            time=detection.timestamp,
                            target_id=detections.target_id,
                            track_id=track_id,
                        )
                    )

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
