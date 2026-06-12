import abc
from dataclasses import dataclass

import numpy as np

from theia.types import DirectShot


class AbstractDamageModel(abc.ABC):
    @abc.abstractmethod
    def is_lethal(self, shot: DirectShot) -> bool:
        raise NotImplementedError()


@dataclass
class RandomDamageModel(AbstractDamageModel):
    p_kill: float
    rng: np.random.Generator

    def is_lethal(self, shot: DirectShot) -> bool:
        return self.rng.uniform() <= self.p_kill
