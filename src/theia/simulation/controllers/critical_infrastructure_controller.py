import datetime

from theia.config import SIDC
from theia.types import (
    ConstantRcsModel,
    Controller,
    Event,
    Point,
    SituationalPicture,
    Target,
    Velocity,
)


class CriticalInfrastructureController(Controller):
    """
    Controller representing a piece of stationary critical infrastructure
    (e. g. an airport, power plant, or command building) that exists purely
    as a target to be defended or attacked - it carries no sensor or
    effector of its own.
    """

    target_id: int
    name: str
    point: Point
    sidc: SIDC
    rcs: float
    """Radar cross section [m^2]"""

    def on_event(self, event: Event):
        pass

    def update(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        self.targets = [
            Target(
                id=self.target_id,
                is_stationary=True,
                name=self.name,
                sidc=self.sidc,
                point=self.point,
                cross_section_model=ConstantRcsModel(rcs=self.rcs),
                velocity=Velocity(vx=0, vy=0, vz=0),
            )
        ]
