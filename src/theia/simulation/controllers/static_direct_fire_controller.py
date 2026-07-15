from dataclasses import dataclass, field
import datetime
from typing import Optional


from theia.config import SIDC
from theia.coordinates import CoordinateTransformations
from theia.coverage import calculate_coverage
from theia.effectors import DirectFireEffector
from theia.types import (
    AbstractEffector,
    ConstantRcsModel,
    Controller,
    Event,
    MonostaticSensor,
    PclSensor,
    Point,
    Receiver,
    SituationalPicture,
    Target,
    Velocity,
)


@dataclass
class StaticDirectFireController(Controller):
    """
    Controller representing a static (i. e. non-moving) direct fire effector.

    A real-world example for such an effector is the Centurion C-RAM.

    This controller attacks only the assigned track.

    Notes
    -----
    No checks are performed whether the effector has attacks left (enough ammo etc.)
    are whether the track is within range. These checks are to be performed by
    the effector during the fire call (no duplicate logic). It is possible that
    the suggested attack is not possible.
    """

    target_id: int
    sidc: SIDC
    rcs: float
    """Radar cross section [m^2]"""
    effector: DirectFireEffector
    assigned_track_id: Optional[str] = None
    """
    Track ID of the track to be fought. No track is fought if ``None``.
    """
    target_name: str = ""
    geojson_range_altitudes: list[float] = field(default_factory=list)

    def __post_init__(self):
        super().__init__()

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
                name=self.target_name,
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
        return []

    def get_firing_effectors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[tuple[AbstractEffector, Point]]:
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

        x, vx, y, vy, z, vz = track(situational_picture.time)
        lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)

        return [(self.effector, Point(lat=lat, lon=lon, alt=alt))]

    def get_geojson(self):
        result = {}
        for alt in self.geojson_range_altitudes:
            coverage = calculate_coverage(
                self.effector.terrain,
                self.effector.point,
                self.effector.combat_range,
                alt,
            )
            result["Effector range @ {alt}MASL"] = coverage
        return result
