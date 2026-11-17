"""
Orders a player gives to their side while the simulation runs.

Actions are only queued when they are received. Whether an action is
possible is decided when it is executed, see
:class:`theia.simulation.controllers.player_controller.PlayerController`.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pydantic

from theia.simulation.path_generators import PathGenerator
from theia.types import Point


class LaunchAction(pydantic.BaseModel):
    """Launch exactly one round from a launch site at a point."""

    type: Literal["launch"] = "launch"
    site_id: int
    """ID of the camp site to launch from"""
    destination: Point
    """Point the round flies to and attacks"""
    path_generator: PathGenerator | None = None
    """Flight path; ``None`` selects the default for the site's ammunition type"""


class MoveStockAction(pydantic.BaseModel):
    """Move ammunition from one camp site to another."""

    type: Literal["move_stock"] = "move_stock"
    source_site_id: int
    target_site_id: int
    count: int = pydantic.Field(gt=0)
    """Number of rounds to move"""


class ToggleMonostaticRadarAction(pydantic.BaseModel):
    """Switch a monostatic radar on or off."""

    type: Literal["toggle_monostatic_radar"] = "toggle_monostatic_radar"
    sensor_id: int
    new_is_on: bool


Action = Annotated[
    LaunchAction | MoveStockAction | ToggleMonostaticRadarAction,
    pydantic.Field(discriminator="type"),
]
