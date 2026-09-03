import datetime
from typing import Optional

from theia.config import SIDC
from theia.coordinates import CoordinateTransformations
from theia.coverage import calculate_coverage
from theia.effectors import DirectFireEffector, IndirectFireEffector
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


class StaticGbadController(Controller):
    """
    Controller representing a static (i. e. non-moving) ground-based air
    defence (GBAD) effector - either direct fire (a real-world example being
    the Centurion C-RAM) or indirect fire, which launches a self-homing
    projectile rather than resolving damage immediately (real-world examples
    being the IRIS-T SL or the MIM-104 Patriot launcher families).

    This controller attacks only the assigned track.

    Notes
    -----
    No checks are performed whether the effector has attacks left (enough ammo
    etc.), whether the track is within range, or whether cadence allows another
    shot yet. These checks are to be performed by the effector during the fire
    call (no duplicate logic). It may happen that the suggested attack is not
    possible.

    Direct fire additionally requires line-of-sight to the target to be
    offered as a shot; indirect fire does not (see
    ``theia.effectors.IndirectFireEffector``).
    """

    target_id: int
    sidc: SIDC
    rcs: float
    """Radar cross section [m^2]"""
    effector: DirectFireEffector | IndirectFireEffector
    assigned_track_id: Optional[str] = None
    """
    Track ID of the track to be fought. No track is fought if ``None``.
    """
    target_name: str = ""
    geojson_range_altitudes: list[float] = []

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
                velocity=Velocity(vx=0, vy=0, vz=0),
                receiver=None,
                transmitter=None,
            )
        ]

        # Fire.
        self.firing_effectors = []
        if self.assigned_track_id is not None:
            track = next(
                (
                    track
                    for track in situational_picture.enemy_targets
                    if track.id == self.assigned_track_id
                ),
                None,
            )
            if track is not None:
                # Queried at situational_picture.time, not +dt.
                # See "Fight before updating the world" in Simulator.advance.
                x, vx, y, vy, z, vz = track(situational_picture.time)
                lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(
                    x, y, z
                )
                target_position = Point(lat=lat, lon=lon, alt=alt)

                if isinstance(self.effector, DirectFireEffector):
                    if self.effector.terrain.has_line_of_sight(
                        self.effector.point, target_position
                    ):
                        self.firing_effectors = [(self.effector, target_position)]
                else:
                    self.effector.assigned_track_id = self.assigned_track_id
                    self.firing_effectors = [(self.effector, target_position)]

        # GeoJSON.
        geojson = {}
        if isinstance(self.effector, DirectFireEffector):
            for alt in self.geojson_range_altitudes:
                coverage = calculate_coverage(
                    self.effector.terrain,
                    self.effector.point,
                    self.effector.combat_range,
                    alt,
                )
                geojson[f"Effector range @ {alt}MASL"] = GeoJSONFeature(
                    geometry=GeoJSONPolygon.from_shapely(coverage),
                    properties={"name": "my polygon"},
                )
        self.geojson = geojson
