import datetime
from dataclasses import dataclass, field
from typing import Optional

from theia.config import SIDC
from theia.coordinates import CoordinateTransformations
from theia.coverage import calculate_coverage
from theia.effectors import DirectFireEffector
from theia.types import (
    ConstantRcsModel,
    Controller,
    Event,
    GeoJSONFeature,
    GeoJSONPolygon,
    Point,
    SituationalPicture,
    Target,
    Velocity,
)


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
    cadence: float
    """Number of attacks per second"""
    time_of_last_shot: Optional[datetime.datetime] = None
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

    def update(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        # Targets.
        self.targets = [
            Target(
                id=self.target_id,
                is_stationary=True,
                name=self.target_name,
                sidc=self.sidc
                if self.effector.n_attacks_left > 0
                else SIDC.damaged(self.sidc.value),
                point=self.effector.point,
                cross_section_model=ConstantRcsModel(rcs=self.rcs),
                velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
            )
        ]

        # Fire.
        self.firing_effectors = []
        ready_to_fire = (
            self.time_of_last_shot is None
            or (situational_picture.time - self.time_of_last_shot).seconds
            < 1 / self.cadence
        )
        if self.assigned_track_id is not None and ready_to_fire:
            track = next(
                (
                    t
                    for t in situational_picture.enemy_targets
                    if t.id == self.assigned_track_id
                ),
                None,
            )
            if track is not None:
                x, vx, y, vy, z, vz = track(situational_picture.time)
                lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
                target_position = Point(lat=lat, lon=lon, alt=alt)

                if self.effector.terrain.has_line_of_sight(
                    self.effector.point, target_position
                ):
                    self.time_of_last_shot = situational_picture.time + dt
                    self.firing_effectors = [(self.effector, target_position)]

        # GeoJSON.
        geojson = {}
        for alt in self.geojson_range_altitudes:
            coverage = calculate_coverage(
                self.effector.terrain,
                self.effector.point,
                self.effector.combat_range,
                alt,
            )
            coverage = GeoJSONFeature(
                geometry=GeoJSONPolygon.from_shapely(coverage),
                properties={"name": "my polygon"},
            )
            geojson["Effector range @ {alt}MASL"] = coverage
        self.geojson = geojson
