import datetime
from dataclasses import dataclass
from typing import Optional

import pydantic

from theia.config import SIDC, UNKNOWN_ID, UNKNOWN_TIME
from theia.distance import line_of_sight_distance
from theia.terrain import AbstractTerrainModel
from theia.types import (
    AbstractEffector,
    ConcreteRcsModel,
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


class OnCooldownException(TheiaException):
    # Cadence has not elapsed since the last confirmed shot.
    pass


class TooManyInFlightException(TheiaException):
    # max_in_flight projectiles from this effector are already in flight.
    pass


def _check_cadence(effector: AbstractEffector, time: datetime.datetime):
    if (
        effector.time_of_last_shot is not None
        and (time - effector.time_of_last_shot).total_seconds()
        < 1 / effector.cadence
    ):
        raise OnCooldownException(
            f"Effector #{effector.id} is still on cooldown "
            f"(last shot at {effector.time_of_last_shot}, cadence {effector.cadence})."
        )


class DirectFireEffector(pydantic.BaseModel, AbstractEffector):
    terrain: AbstractTerrainModel

    def __deepcopy__(self, memo=None):
        """
        ``terrain`` is the shared, read-only world model. It also holds
        thread-local LOS scratch buffers (see ``HbvTree._tls``) that cannot
        be deep-copied. Cloning an effector (e.g. instantiating a projectile
        from its template) must keep the same terrain reference rather than
        duplicating the world.
        """
        if memo is None:
            memo = {}
        memo[id(self.terrain)] = self.terrain
        return super().__deepcopy__(memo)

    def fire(self, target: Target, time: datetime.datetime) -> DirectShot:
        """
        Raises
        ------
        OnCooldownException:
            Cadence period has not elapsed since the last confirmed shot
        OutOfAttacksException:
            There are no attacks left
        OutOfRangeException:
            The target is not within range
        NoLosException:
            There is no line-of-sight to the target
        """
        _check_cadence(self, time)
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
        self.time_of_last_shot = time

        return DirectShot(
            id=UNKNOWN_ID,
            time=UNKNOWN_TIME,
            shooter=self,
            target=target,
        )


@dataclass
class IndirectFireEffector(AbstractEffector):
    """
    Launches a self-homing projectile rather than resolving damage immediately.

    Unlike ``DirectFireEffector``, line-of-sight to the target is deliberately
    not required to fire: indirect fire (e. g. artillery, SAM launchers) can
    engage targets it cannot itself see.
    """

    projectile: DirectFireEffector
    """
    Template for the launched projectile
    (e. g. warheads, loitering ammo, interceptor drone, ...)
    """
    projectile_speed: float
    """Cruise speed of the launched projectile [m / s]"""
    projectile_max_dist: float
    """Maximum distance the launched projectile can travel [m]"""
    projectile_sidc: SIDC
    projectile_rcs: ConcreteRcsModel
    projectile_name: str = ""
    assigned_track_id: Optional[str] = None
    """
    ID of the track this launch is aimed at.
    """
    max_in_flight: int = 1
    """
    Maximum number of this effector's own projectiles allowed to be in
    flight (launched but not yet resolved) at the same time.
    """
    n_in_flight: int = 0
    """
    Number of this effector's projectiles currently in flight. Incremented
    on a confirmed launch; decremented by the projectile's controller
    (``HomingSystem``) once it is destroyed, for whatever reason (hit,
    lost track, out of fuel).
    """

    def fire(self, target: Target, time: datetime.datetime) -> IndirectShot:
        """
        Raises
        ------
        OnCooldownException:
            Cadence period has not elapsed since the last confirmed shot
        OutOfAttacksException:
            There are no attacks left
        OutOfRangeException:
            The target is not within range
        TooManyInFlightException:
            ``max_in_flight`` of this effector's projectiles are already
            in the air
        """
        _check_cadence(self, time)
        if self.n_in_flight >= self.max_in_flight:
            raise TooManyInFlightException(
                f"Effector #{self.id} already has {self.n_in_flight} "
                f"projectile(s) in flight (max {self.max_in_flight})."
            )
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
        self.time_of_last_shot = time
        self.n_in_flight += 1

        return IndirectShot(
            id=UNKNOWN_ID,
            time=UNKNOWN_TIME,
            shooter=self,
            target=target,
            projectile=self.projectile,
            track_id=self.assigned_track_id,
        )
