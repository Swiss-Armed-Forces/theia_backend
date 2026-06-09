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
from theia.config import SIDC_UNKNOWN
from theia.measurement import MonostaticMeasurementTransformations
from theia.stonesoup_interface import MonostaticDetectionFactory
from theia.types import (
    CLUTTER_TARGET,
    AbstractTracker,
    MonostaticRadarDetection,
    PclDetection,
    PetDetection,
)


class DummyTracker(AbstractTracker):
    def add_detections(
        self,
        monostatic_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
    ):
        pass

    def get_tracks(self) -> list[theia.types.Track]:
        return []
