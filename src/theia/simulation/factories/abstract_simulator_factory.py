import abc
import datetime

import numpy as np

from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.damage_model import AbstractDamageModel
from theia.simulation.logging import SituationalPictureBuffer
from theia.simulation.simulator import Simulator, TerminationCriterion
from theia.terrain import AbstractTerrainModel
from theia.types import AbstractTracker, Controller


class AbstractSimulatorFactory(abc.ABC):
    @abc.abstractmethod
    def _get_pcl_detector(self) -> PclDetector:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_pet_detector(self) -> PetDetector:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_blue_tracker(self) -> AbstractTracker:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_red_tracker(self) -> AbstractTracker:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_blue_controller(self) -> Controller:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_red_controller(self) -> Controller:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_start_time(self) -> datetime.datetime:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_timestep(self) -> datetime.timedelta:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_termination_criterion(self) -> TerminationCriterion:
        raise NotImplementedError()

    def build_simulator(
        self,
        interactive: bool,
        rng: np.random.Generator,
        terrain_model: AbstractTerrainModel,
        damage_model: AbstractDamageModel,
    ) -> tuple[Simulator, SituationalPictureBuffer]:
        buffer = SituationalPictureBuffer()
        simulator = Simulator(
            pcl_detector=self._get_pcl_detector(),
            pet_detector=self._get_pet_detector(),
            blue_controller=self._get_blue_controller(),
            red_controller=self._get_red_controller(),
            blue_tracker=self._get_blue_tracker(),
            red_tracker=self._get_red_tracker(),
            start_time=self._get_start_time(),
            time_step=self._get_timestep(),
            min_time_per_step=datetime.timedelta(seconds=1 if interactive else 0),
            termination_criterion=self._get_termination_criterion(),
            rng=np.random.Generator(np.random.PCG64(seed=4054080)),
            listener=buffer,
            simulate_clutter=False,
            terrain_model=terrain_model,
            damage_model=damage_model,
        )
        return simulator, buffer
