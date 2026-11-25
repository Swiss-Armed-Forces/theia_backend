import datetime
from typing import Optional

import pydantic
import shapely

from theia.coordinates import CoordinateTransformations
from theia.coverage import calculate_coverage, calculate_range_polygon
from theia.distance import burstvincentydistance, line_of_sight_distance
from theia.effectors import DirectFireEffector, IndirectFireEffector
from theia.types import (
    ConstantRcsModel,
    Controller,
    Event,
    GeoJSONAnnotation,
    GeoJSONFeature,
    GeoJSONKind,
    GeoJSONMultiPolygon,
    Point,
    SituationalPicture,
    Target,
    TargetInfo,
    Velocity,
)


class StaticGbadController(Controller):
    """
    Controller representing a static (i. e. non-moving) ground-based air
    defence (GBAD) effector - either direct fire (a real-world example being
    the Centurion C-RAM) or indirect fire, which launches a self-homing
    projectile rather than resolving damage immediately (real-world examples
    being the IRIS-T SL or the MIM-104 Patriot launcher families).

    This controller attacks only the assigned track, and only if that track
    is airborne (``track.target_info.is_airborne``); otherwise no shot is
    offered. A GBAD cannot engage ground targets (e.g. a Patriot launcher
    cannot engage a fighter jet on the ground).

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
    info: TargetInfo
    rcs: float
    """Radar cross section [m^2]"""
    effector: DirectFireEffector | IndirectFireEffector
    assigned_track_id: Optional[str] = None
    """
    Track ID of the track to be fought. No track is fought if ``None``.
    """
    target_name: str = ""
    geojson_range_altitudes: set[int] = set()
    """Altitudes at which the effector range is drawn [m above sea level]"""

    _range_cache: dict[int, GeoJSONMultiPolygon] = pydantic.PrivateAttr(
        default_factory=dict
    )
    """Effector range per altitude; static, because the GBAD does not move"""

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
                info=self.info,
                is_damaged=self.effector.n_attacks_left <= 0,
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
            # Never engage a track on the ground (see class docstring).
            if track is not None and track.target_info.is_airborne:
                # Queried at situational_picture.time, not +dt.
                # See "Fight before updating the world" in Simulator.advance.
                x, vx, y, vy, z, vz = track(situational_picture.time)
                lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
                target_position = Point(lat=lat, lon=lon, alt=alt)

                if isinstance(self.effector, DirectFireEffector):
                    if self.effector.terrain.has_line_of_sight(
                        self.effector.point, target_position
                    ):
                        self.firing_effectors = [(self.effector, target_position)]
                elif isinstance(self.effector, IndirectFireEffector):
                    self.effector.assigned_track_id = self.assigned_track_id
                    self.firing_effectors = [(self.effector, target_position)]
                else:
                    raise TypeError(
                        f"Unknown effector type {type(self.effector).__name__}"
                    )

        # GeoJSON.
        geojson: list[GeoJSONAnnotation] = []
        name = self.target_name or self.effector.name
        for alt in sorted(self.geojson_range_altitudes):
            effector_range = self._effector_range(alt)
            if not effector_range.coordinates:
                continue  # Altitude out of range
            geojson.append(
                GeoJSONAnnotation(
                    id=f"effector-range-{self.target_id}-{alt}",
                    feature=GeoJSONFeature(geometry=effector_range),
                    kind=GeoJSONKind.EFFECTOR_RANGE,
                    target_ids=[self.target_id],
                    hover_text=f"{name} range @ {alt} MASL",
                )
            )
        self.geojson = geojson

    def _effector_range(self, alt: int) -> GeoJSONMultiPolygon:
        """
        Area at `alt` [m above sea level] in which a target can be engaged.

        Direct fire: within combat range and line of sight. Indirect fire:
        within combat range only, as checked by ``IndirectFireEffector.fire``,
        i. e. a circle. Empty if `alt` is out of range.
        """
        if alt not in self._range_cache:
            if isinstance(self.effector, DirectFireEffector):
                polygons = calculate_coverage(
                    self.effector.terrain,
                    self.effector.point,
                    self.effector.combat_range,
                    alt,
                )
            elif isinstance(self.effector, IndirectFireEffector):
                radius = self._horizontal_range(alt)
                polygons = (
                    [calculate_range_polygon(self.effector.point, radius, alt)]
                    if radius > 0
                    else []
                )
            else:
                raise TypeError(
                    f"Unknown effector type {type(self.effector).__name__}"
                )
            # The coverage may fall apart into several polygons.
            self._range_cache[alt] = GeoJSONMultiPolygon.from_shapely(
                shapely.MultiPolygon(polygons)
            )
        return self._range_cache[alt]

    def _horizontal_range(self, alt: int) -> float:
        """
        Geodesic distance [m] at which a target at `alt` [m above sea level]
        is exactly at combat range (slant distance), or 0 if it is never in
        range. Found by bisection, because the slant distance also depends on
        the Earth's curvature.
        """
        p = self.effector.point
        combat_range = self.effector.combat_range

        def slant_distance(d: float) -> float:
            q = burstvincentydistance((p.lat, p.lon), d, 0, alt)
            return line_of_sight_distance(p.lat, p.lon, p.alt, q.lat, q.lon, alt)

        if slant_distance(0) > combat_range:
            return 0.0
        # The slant distance may be shorter than the geodesic distance (chord).
        lo, hi = 0.0, combat_range
        while slant_distance(hi) <= combat_range:
            lo, hi = hi, 2 * hi
        while hi - lo > 1.0:
            mid = (lo + hi) / 2
            if slant_distance(mid) <= combat_range:
                lo = mid
            else:
                hi = mid
        return lo
