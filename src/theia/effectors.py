from dataclasses import dataclass

from theia.config import UNKNOWN_ID, UNKNOWN_TIME
from theia.distance import line_of_sight_distance
from theia.terrain import AbstractTerrainModel
from theia.types import AbstractEffector, DirectShot, IndirectShot, Target


@dataclass
class DirectFireEffector(AbstractEffector):
    terrain: AbstractTerrainModel

    def fire(self, target: Target) -> DirectShot:
        d = line_of_sight_distance(
            self.point.lat,
            self.point.lon,
            self.point.alt,
            target.lat,
            target.lon,
            target.alt,
        )
        if self.n_attacks_left <= 0:
            raise ValueError("No attack left.")
        if d > self.combat_range:
            raise ValueError("Target out of range.")
        if not self.terrain.has_line_of_sight(self.point, target.point):
            raise ValueError("No line-of-sight to target.")

        self.n_attacks_left -= 1

        return DirectShot(
            id=UNKNOWN_ID,
            time=UNKNOWN_TIME,
            shooter=self,
            target=target,
        )


@dataclass
class IndirectFireEffector(AbstractEffector):
    projectile: DirectFireEffector

    def fire(self, target: Target) -> IndirectShot:
        d = line_of_sight_distance(
            self.point.lat,
            self.point.lon,
            self.point.alt,
            target.lat,
            target.lon,
            target.alt,
        )
        if self.n_attacks_left <= 0 or d > self.combat_range:
            raise ValueError("No attack left.")

        self.n_attacks_left -= 1

        return IndirectShot(
            id=UNKNOWN_ID,
            time=UNKNOWN_TIME,
            shooter=self,
            target=target,
            projectile=self.projectile,
        )
