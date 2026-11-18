"""
Orders an actor (e. g. a player or an AI) gives to its side while the
simulation runs.

Actions are only queued when they are received. Whether an action is
possible is decided when it is executed, see
:class:`theia.simulation.controllers.action_controller.ActionController`.

Actions are events: the ``ActionController`` logs every action it executes,
followed by an :class:`ActionExecutedEvent` or an :class:`ActionRejectedEvent`
that refers to it by ``action_id``.
"""

from __future__ import annotations

import datetime
import enum
from typing import Annotated, Literal

import pydantic

from theia.config import UNKNOWN_ID, UNKNOWN_TIME
from theia.simulation.path_generators import PathGenerator
from theia.types import Event, Party, Point


class Action(Event):
    """
    Base class of all actions.

    The actor only gives the order. ``id``, ``time`` and ``party`` are filled
    in by the ``ActionController`` when it executes the action; values given
    by the actor are overwritten.
    """

    id: int = UNKNOWN_ID
    time: datetime.datetime = UNKNOWN_TIME
    party: Party = Party.UNKNOWN
    """Side that gave the order"""

    def __str__(self) -> str:
        order = self.model_dump_json(exclude={"id", "time", "party"})
        return f"[ACTION {self.party.value}] #{self.id}: {order}"


class LaunchAction(Action):
    """Launch exactly one round from a launch site at a point."""

    type: Literal["launch"] = "launch"
    site_id: int
    """ID of the camp site to launch from"""
    destination: Point
    """Point the round flies to and attacks"""
    path_generator: PathGenerator | None = None
    """Flight path; ``None`` selects the default for the site's ammunition type"""


class MoveStockAction(Action):
    """Move ammunition from one camp site to another."""

    type: Literal["move_stock"] = "move_stock"
    source_site_id: int
    target_site_id: int
    count: int = pydantic.Field(gt=0)
    """Number of rounds to move"""


class ToggleMonostaticRadarAction(Action):
    """Switch a monostatic radar on or off."""

    type: Literal["toggle_monostatic_radar"] = "toggle_monostatic_radar"
    sensor_id: int
    new_is_on: bool


AnyAction = Annotated[
    LaunchAction | MoveStockAction | ToggleMonostaticRadarAction,
    pydantic.Field(discriminator="type"),
]
"""Any concrete action, told apart by ``type`` (for parsing requests and logs)"""


class ActionExecutedEvent(Event):
    """Fired after an action has been executed."""

    party: Party
    action_id: int
    """``id`` of the executed :class:`Action`"""
    result: str
    """What the action did"""

    def __str__(self) -> str:
        return f"[ACTION {self.party.value}] #{self.action_id} executed: {self.result}"


class RejectionReason(enum.Enum):
    """Why an action could not be executed."""

    UNKNOWN_ACTION_TYPE = "UNKNOWN_ACTION_TYPE"
    UNKNOWN_CAMP_SITE = "UNKNOWN_CAMP_SITE"
    CAMP_SITE_DESTROYED = "CAMP_SITE_DESTROYED"
    CANNOT_LAUNCH = "CANNOT_LAUNCH"
    """The camp site is a depot."""
    NO_STOCK = "NO_STOCK"
    """The launch site has no rounds left."""
    RELOADING = "RELOADING"
    """The launch site has launched recently and is not ready yet."""
    NO_FLIGHT_PATH = "NO_FLIGHT_PATH"
    """No flight path to the destination exists."""
    NO_ROUTE = "NO_ROUTE"
    """There is no transport route between the two camp sites."""
    INSUFFICIENT_STOCK = "INSUFFICIENT_STOCK"
    """The source site has fewer rounds than are to be moved."""
    UNKNOWN_RADAR = "UNKNOWN_RADAR"
    RADAR_DESTROYED = "RADAR_DESTROYED"


class ActionRejectedEvent(Event):
    """Fired instead of :class:`ActionExecutedEvent` if an action is impossible."""

    party: Party
    action_id: int
    """``id`` of the rejected :class:`Action`"""
    reason: RejectionReason
    message: str
    """Human-readable details, e.g. the names of the sites involved"""

    def __str__(self) -> str:
        return (
            f"[ACTION {self.party.value}] #{self.action_id} rejected "
            f"({self.reason.value}): {self.message}"
        )
