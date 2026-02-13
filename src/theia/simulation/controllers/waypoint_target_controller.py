import datetime
from theia.types import SituationalPicture, Target, TargetController, Trajectory


class WaypointTargetController(TargetController):
    def __init__(self, trajectories: list[Trajectory]):
        self._trajectories = trajectories

    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Target]:
        targets = [t(situational_picture.time + dt) for t in self._trajectories]
        targets = [target for target in targets if target is not None]
        return targets
