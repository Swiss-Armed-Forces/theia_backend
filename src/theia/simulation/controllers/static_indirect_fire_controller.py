import datetime
from typing import Optional

from theia.config import SIDC
from theia.coordinates import CoordinateTransformations
from theia.coverage import calculate_coverage
from theia.effectors import IndirectFireEffector
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


class StaticIndirectFireController(Controller):
    """
    Controller representing a static (i. e. non-moving) indirect fire effector.
    Indirect fire means that instead of attacking the target immediately like
    with small calibre weapons, a projectile is deployed.

    
    A real-world example for such an effector is the IRIS-T SL or the
    MIM-104 Patriot launcher families.

    This controller attacks only the assigned track.

    Notes
    -----
    No checks are performed whether the effector has attacks left (enough ammo
    etc.), whether the track is within range, or whether cadence allows another
    launch yet. These checks are to be performed by the effector during the
    fire call (no duplicate logic). It is possible that the suggested attack is
    not possible.
    """

    target_id: int
    sidc: SIDC
    rcs: float
    """Radar cross section [m^2]"""
    effector: IndirectFireEffector
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

        # Launch.
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
                # Queried at situational_picture.time (not +dt): _execute_attacks
                # matches this aim point against ground-truth targets that are
                # one tick stale (see "Fight before updating the world" in
                # Simulator.advance), so the aim point must be computed on that
                # same, un-advanced time basis to actually line up with it.
                x, vx, y, vy, z, vz = track(situational_picture.time)
                lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(
                    x, y, z
                )
                target_position = Point(lat=lat, lon=lon, alt=alt)
                self.effector.assigned_track_id = self.assigned_track_id
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
