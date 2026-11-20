import datetime

import pydantic

from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.intel_controller import (
    IntelController,
    build_stationary_intel_sensor,
)
from theia.types import (
    Controller,
    DirectShot,
    Event,
    IndirectShot,
    Party,
    Shot,
    SituationalPicture,
    TargetInfos,
)


class ShooterRevealController(ControllerGroup):
    """
    Reveals the positions of enemy shooters to the observing side.

    For each shot fired by the enemy, one `IntelController` child is added and
    kept for `valid_time` after the shot. It exposes an `IntelSensor` that
    reports the shooter's position. Repeated shots by the same
    shooter each add their own child. Own shooters' positions are known
    perfectly, so own shots are ignored.

    Enemy-shot rule: the simulator always aims a shot at a target of the
    shooter's enemy (see `Simulator._execute_attacks`). Hence a shot whose
    target belongs to the observing side was fired by the other side. Shots at
    targets of party `UNKNOWN` or `NEUTRAL` cannot be attributed and are
    ignored.

    Lifecycle: the expiry time (`shot time + valid_time`) of each child is
    stored here, not in the child. At the start of `update`, children whose
    expiry is at or before the situational picture's time are removed.

    The IDs of the revealed targets and their intel sensors come from the
    simulation's `IdProvider` (see `Controller.simulation_context`), so the
    controller must be bound to a simulation before it receives a shot.
    """

    is_blue: bool
    """Side that observes the shots."""
    valid_time: datetime.timedelta
    """How long the knowledge about a shooter remains after a shot."""

    _expiries: list[tuple[Controller, datetime.datetime]] = pydantic.PrivateAttr(
        default_factory=list
    )
    """Children created by this controller and their expiry times."""

    def __init__(self, *, is_blue: bool, valid_time: datetime.timedelta):
        super().__init__([], is_blue=is_blue, valid_time=valid_time)

    def _is_enemy_shot(self, event: Event) -> bool:
        if not isinstance(event, (DirectShot, IndirectShot)):
            return False
        # A shot is always aimed at the shooter's enemy, so a shot at one of
        # our own targets was fired by the enemy.
        return event.target.info.party == Party.from_is_blue(self.is_blue)

    def on_event(self, event: Event):
        if self._is_enemy_shot(event):
            shot: Shot = event
            sensor = build_stationary_intel_sensor(
                self.simulation_context.id_provider,
                shot.shooter.point,
                TargetInfos.UNKNOWN,
            )
            child = IntelController(sensor=sensor)
            self.add_controller(child)
            self._expiries.append((child, event.time + self.valid_time))
        super().on_event(event)

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        now = situational_picture.time
        expired_ids = {id(c) for c, expiry in self._expiries if expiry <= now}
        if expired_ids:
            # Compare by identity: stateless children are equal by value.
            self._expiries = [
                (c, expiry) for c, expiry in self._expiries if id(c) not in expired_ids
            ]
            self._controllers = [
                c for c in self._controllers if id(c) not in expired_ids
            ]
        super().update(situational_picture, dt)
