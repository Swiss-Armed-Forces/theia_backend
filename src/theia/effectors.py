from dataclasses import dataclass

from theia.config import UNKNOWN_ID, UNKNOWN_TIME
from theia.distance import line_of_sight_distance
from theia.terrain import AbstractTerrainModel
from theia.types import (
    AbstractEffector,
    DirectShot,
    IndirectShot,
    Target,
    TheiaException,
)


class OutOfAttacksException(TheiaException):
    pass


class OutOfRangeException(TheiaException):
    pass


class NoLosException(TheiaException):
    # No line of sight.
    pass


@dataclass
class DirectFireEffector(AbstractEffector):
    terrain: AbstractTerrainModel

    def fire(self, target: Target) -> DirectShot:
        """
        Raises
        ------
        OutOfAttacksException:
            There are no attacks left
        OutOfRangeException:
            The target is not within range
        NoLosException:
            There is no line-of-sight to the target
        """
        d = line_of_sight_distance(
            self.point.lat,
            self.point.lon,
            self.point.alt,
            target.lat,
            target.lon,
            target.alt,
        )
        if self.n_attacks_left <= 0:
            raise OutOfAttacksException("No attack left.")
        if d > self.combat_range:
            raise OutOfRangeException(
                f"Target at {target.point} out of range ({self.combat_range:.1f} m)."
            )
        if not self.terrain.has_line_of_sight(self.point, target.point):
            raise NoLosException(f"No line-of-sight to target at {target.point}.")

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
        if self.n_attacks_left <= 0:
            raise OutOfAttacksException("No attack left.")
        if d > self.combat_range:
            raise OutOfRangeException(
                f"Target at {target.point} out of range ({self.combat_range:.1f} m)."
            )

        self.n_attacks_left -= 1

        return IndirectShot(
            id=UNKNOWN_ID,
            time=UNKNOWN_TIME,
            shooter=self,
            target=target,
            projectile=self.projectile,
        )
