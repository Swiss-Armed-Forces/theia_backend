from dataclasses import dataclass
import datetime

from theia.config import UNKNOWN_ID
from theia.distance import line_of_sight_distance
from theia.terrain import AbstractTerrainModel
from theia.types import (
    AbstractEffector,
    Controller,
    KillEvent,
    Point,
    Shot,
    SituationalPicture,
    Trajectory,
)


@dataclass
class FixedPathOneWayDrone(Controller):
    """
    Represent a drone that travels on a predefined trajectory and attacks a
    target once it is within the attack radius of the destination.
    """

    effector: AbstractEffector
    trajectory: Trajectory
    assigned_goal: Point
    terrain: AbstractTerrainModel

    def __post_init__(self):
        super().__init__()

    def on_event(self, event):
        if isinstance(event, Shot) and event.shooter.id == self.effector.id:
            # Commit suicide.
            self._broadcast_event(
                KillEvent(
                    id=UNKNOWN_ID,
                    time=event.time,
                    target_id=self.trajectory.target_id,
                )
            )

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        target = self.trajectory(situational_picture.time)
        self.targets = [] if target is None else [target]

        if target is None:
            self.firing_effectors = []
            return

        p = target.point
        if (
            self.terrain.has_line_of_sight(p, self.assigned_goal)
            and line_of_sight_distance(*p.as_tuple(), *self.assigned_goal.as_tuple())
            <= self.effector.combat_range
        ):
            self.effector.point = p
            self.firing_effectors = [(self.effector, self.assigned_goal)]
        else:
            return []
