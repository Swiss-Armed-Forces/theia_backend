"""
Controller of a camp site during the game.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from theia.types import (
    ConstantRcsModel,
    Controller,
    Event,
    SituationalPicture,
    Target,
    Velocity,
)

if TYPE_CHECKING:
    # Not imported at runtime: scenario_import imports this module. The model
    # is rebuilt in scenario_import once CampSite is defined.
    from theia.simulation.scenario_import import CampSite


class CampSiteController(Controller):
    """
    Camp site of a side during the game.

    The site is a target. Its ``camp_site.n_ammo`` is the current stock, which
    the ``PlayerController`` changes by launches and shipments. The controller
    must own its ``CampSite`` (pass a copy) so that the order of battle keeps
    its initial stock. Wrap it in a ``LivingController`` to make it killable.
    """

    camp_site: CampSite
    is_blue: bool
    time_of_last_launch: datetime.datetime | None = None

    def seconds_until_ready(self, time: datetime.datetime) -> float:
        """Remaining reload time before the next launch [s]; 0 means ready."""
        if self.time_of_last_launch is None:
            return 0.0
        elapsed = (time - self.time_of_last_launch).total_seconds()
        return max(0.0, self.camp_site.seconds_between_launches - elapsed)

    def on_event(self, event: Event):
        pass

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        self.targets = [self._target()]

    def _target(self) -> Target:
        site = self.camp_site
        return Target(
            id=site.target_id,
            is_stationary=True,
            name=site.name,
            info=site.target_info(self.is_blue),
            point=site.point,
            cross_section_model=ConstantRcsModel(rcs=site.rcs),
            velocity=Velocity(vx=0, vy=0, vz=0),
        )
