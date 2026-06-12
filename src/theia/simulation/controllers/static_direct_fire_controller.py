import datetime
from typing import Optional

import pydantic

from theia.config import SIDC
from theia.effectors import DirectFireEffector
from theia.types import (
    ConstantRcsModel,
    Controller,
    DirectShot,
    Event,
    MonostaticSensor,
    PclSensor,
    Receiver,
    SituationalPicture,
    Target,
    Velocity,
)


class StaticDirectFireController(Controller, pydantic.BaseModel):
    """
    Controller representing a static (i. e. non-moving) direct fire effector.

    A real-world example for such an effector is the Centurion C-RAM.
    """

    target_id: int
    sidc: SIDC
    rcs: float
    """Radar cross section [m^2]"""
    effector: DirectFireEffector
    assigned_track_id: Optional[int] = None
    """Track ID of the track to be fought"""

    def on_event(self, event: Event):
        pass

    def get_monostatic_radars(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[MonostaticSensor]:
        return []

    def get_pcl_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[PclSensor]:
        return []

    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Target]:
        return [
            Target(
                id=self.target_id,
                is_stationary=True,
                name=(f"{self.effector.name} (Target)"),
                sidc=self.sidc,
                point=self.effector.point,
                cross_section_model=ConstantRcsModel(rcs=self.rcs),
                velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
            )
        ]

    def get_pet_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        raise NotImplementedError

    def get_shots(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[DirectShot]:
        if self.assigned_track_id is None:
            return []
        track = next(
            (
                t
                for t in situational_picture.enemy_targets
                if t.id == self.assigned_track_id
            ),
            None,
        )

        if track is None:
            return []

        return [
            DirectShot(
                id=-1,
                time=situational_picture.time + dt,
                shooter=self.effector,
                track=track,
            )
        ]
