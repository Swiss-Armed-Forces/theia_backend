"""
Strategies that turn a launch (start point, destination, launch time, speed)
into a fixed flight path.

Each generator is a pydantic model, so a player can choose one (and its
parameters) as part of a :class:`theia.simulation.actions.LaunchAction`.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
import pydantic

from theia.config import (
    CM_DEFAULT_CRUISE_MAGL,
    CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE,
    CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE,
    CM_DEFAULT_SAMPLE_SPACING,
    CM_DEFAULT_TERMINAL_DIVE_ANGLE,
    DRONE_DEFAULT_CLIMB_ANGLE,
    DRONE_DEFAULT_CRUISE_ALT_MASL,
    DRONE_DEFAULT_DIVE_START_DISTANCE,
    DRONE_SAMPLE_SPACING,
)
from theia.coordinates import CoordinateTransformations
from theia.cruise_missile import (
    build_terrain_following_path,
    default_min_clearance,
    great_circle_points,
    validate_parameters,
)
from theia.distance import haversine, line_of_sight_distance
from theia.terrain import AbstractTerrainModel
from theia.types import Point


@dataclass
class GeneratedPath:
    """Flight path produced by a :data:`PathGenerator`."""

    times: list[datetime.datetime]
    lats: list[float]
    """Latitudes [°]"""
    lons: list[float]
    """Longitudes [°]"""
    alts: list[float]
    """Altitudes [m above sea level]"""

    @property
    def end_point(self) -> Point:
        """Where the round ends up (the point it attacks)."""
        return Point(lat=self.lats[-1], lon=self.lons[-1], alt=self.alts[-1])


class ClimbCruiseDiveGenerator(pydantic.BaseModel):
    """
    Climb linearly to a cruise altitude, fly along the great circle and dive
    linearly onto the destination.

    Default path generator for one-way drones.
    """

    type: Literal["climb_cruise_dive"] = "climb_cruise_dive"
    cruise_alt_masl: float = DRONE_DEFAULT_CRUISE_ALT_MASL
    """Cruise altitude [m above sea level]"""
    dive_start_distance: float = pydantic.Field(
        default=DRONE_DEFAULT_DIVE_START_DISTANCE, gt=0
    )
    """Distance along the ground track before the destination at which the dive starts [m]"""
    climb_angle: float = pydantic.Field(default=DRONE_DEFAULT_CLIMB_ANGLE, gt=0, lt=90)
    """Flight-path angle of the climb [°]"""

    def profile(self, p_start: Point, p_stop: Point) -> tuple[np.ndarray, np.ndarray]:
        """
        Corners of the piecewise linear altitude profile.

        Returns
        -------
        s: np.ndarray
            Distances along the ground track [m], increasing from 0 to the
            great-circle distance between start and destination.
        z: np.ndarray
            Altitudes [m above sea level] at ``s``.
        """
        distance = haversine(p_start.lon, p_start.lat, p_stop.lon, p_stop.lat)
        if distance <= 0:
            raise ValueError("Start and destination coincide")

        z0 = p_start.alt
        z_end = p_stop.alt
        z_cruise = self.cruise_alt_masl
        tan_climb = math.tan(math.radians(self.climb_angle))
        dive_slope = (z_cruise - z_end) / self.dive_start_distance

        def z_climb(s):
            return z0 + s * tan_climb

        def z_dive(s):
            return z_end + (distance - s) * dive_slope

        def profile(s):
            return min(z_climb(s), z_cruise, z_dive(s))

        # The profile is the minimum of three lines. Its corners are where
        # two of the lines intersect.
        corners = [
            (z_cruise - z0) / tan_climb,  # climb meets cruise
            distance - self.dive_start_distance,  # cruise meets dive
        ]
        if tan_climb + dive_slope != 0:
            # climb meets dive
            # Obtained by setting z_climb(s) == z_dive(s) and solving for s:
            # z0 + s·tan_climb = z_end + (distance − s)·dive_slope
            # s·(tan_climb + dive_slope) = z_end + distance·dive_slope − z0
            # s = (z_end + distance·dive_slope − z0) / (tan_climb + dive_slope)
            corners.append(
                (z_end + distance * dive_slope - z0) / (tan_climb + dive_slope)
            )
        s = np.array(
            [0.0] + sorted(c for c in corners if 0 < c < distance) + [distance]
        )
        z = np.array([profile(x) for x in s])
        if not math.isclose(z[-1], z_end, abs_tol=1e-6):
            raise ValueError(
                f"The destination at {z_end:.0f} MASL is out of reach: "
                f"the path only reaches {z[-1]:.0f} MASL there"
            )
        return s, z

    def generate(
        self,
        p_start: Point,
        p_stop: Point,
        t_start: datetime.datetime,
        speed: float,
        terrain: AbstractTerrainModel,
    ) -> GeneratedPath:
        if speed <= 0:
            raise ValueError(f"speed must be > 0 m/s, got {speed}")
        s_corners, z_corners = self.profile(p_start, p_stop)

        # Sample each linear segment evenly along its 3D length, so that the
        # trajectory's cubic spline hardly overshoots at the corners.
        s_samples = [s_corners[:1]]
        z_samples = [z_corners[:1]]
        for i in range(len(s_corners) - 1):
            ds = s_corners[i + 1] - s_corners[i]
            dz = z_corners[i + 1] - z_corners[i]
            n = max(1, math.ceil(math.hypot(ds, dz) / DRONE_SAMPLE_SPACING))
            f = np.arange(1, n + 1) / n
            s_samples.append(s_corners[i] + f * ds)
            z_samples.append(z_corners[i] + f * dz)
        s = np.concatenate(s_samples)
        z = np.concatenate(z_samples)

        lengths = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(s), np.diff(z)))])
        seconds = lengths / speed
        # Arrive at a whole second after launch, so that a simulation with a
        # 1 s time step sees the drone at its destination.
        seconds[-1] = math.ceil(seconds[-1])

        lats, lons = great_circle_points(p_start, p_stop, s)
        # Avoid round-off at the end points from the coordinate transformation
        # round trip in great_circle_points.
        lats[0], lons[0] = p_start.lat, p_start.lon
        lats[-1], lons[-1] = p_stop.lat, p_stop.lon
        return GeneratedPath(
            times=[t_start + datetime.timedelta(seconds=float(t)) for t in seconds],
            lats=lats.tolist(),
            lons=lons.tolist(),
            alts=z.tolist(),
        )


class BallisticGenerator(pydantic.BaseModel):
    """
    Ballistic trajectory under gravity only, see
    :class:`theia.simulation.scenario_import.BallisticMissileFactory`.

    Default path generator for ballistic missiles. The speed is ignored: it
    follows from the launch angle and the distance.
    """

    type: Literal["ballistic"] = "ballistic"
    alpha: float = pydantic.Field(default=45.0, gt=0, lt=90)
    """Launch angle w. r. t. LOS [°]"""

    def generate(
        self,
        p_start: Point,
        p_stop: Point,
        t_start: datetime.datetime,
        speed: float,
        terrain: AbstractTerrainModel,
    ) -> GeneratedPath:
        _, times, trajectory_data = build_trajectory(
            np.deg2rad(self.alpha), p_start, p_stop
        )
        points = convert_to_geodetic(p_start, p_stop, trajectory_data)
        return GeneratedPath(
            times=[t_start + datetime.timedelta(seconds=float(t)) for t in times],
            lats=[p.lat for p in points],
            lons=[p.lon for p in points],
            alts=[p.alt for p in points],
        )


class TerrainFollowingGenerator(pydantic.BaseModel):
    """
    Terrain-following path, see :mod:`theia.cruise_missile`.

    Default path generator for cruise missiles. The altitudes of the start and
    the destination are ignored, see
    :class:`theia.simulation.scenario_import.CruiseMissileFactory`.
    """

    type: Literal["terrain_following"] = "terrain_following"
    cruise_magl: float = CM_DEFAULT_CRUISE_MAGL
    """Nominal cruise height above ground level [m]"""
    min_flight_path_angle: float = CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE
    """Steepest descent while cruising [°], in (-90°, 0°)"""
    max_flight_path_angle: float = CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE
    """Steepest climb while cruising [°], in (0°, 90°)"""
    terminal_dive_angle: float = CM_DEFAULT_TERMINAL_DIVE_ANGLE
    """Flight-path angle of the terminal dive [°], in (-90°, 0°)"""
    min_clearance: float | None = None
    """Minimum clearance above the terrain while cruising [m].
    ``None`` means ``min(cruise_magl, max(15 m, 0.5 * cruise_magl))``."""
    sample_spacing: float = CM_DEFAULT_SAMPLE_SPACING
    """Distance between trajectory samples along the ground track [m]"""

    @property
    def effective_min_clearance(self) -> float:
        if self.min_clearance is None:
            return default_min_clearance(self.cruise_magl)
        return self.min_clearance

    def generate(
        self,
        p_start: Point,
        p_stop: Point,
        t_start: datetime.datetime,
        speed: float,
        terrain: AbstractTerrainModel,
    ) -> GeneratedPath:
        validate_parameters(
            p_start,
            p_stop,
            speed,
            self.cruise_magl,
            self.min_flight_path_angle,
            self.max_flight_path_angle,
            self.terminal_dive_angle,
            self.effective_min_clearance,
            self.sample_spacing,
        )
        path = build_terrain_following_path(
            p_start=p_start,
            p_stop=p_stop,
            speed=speed,
            cruise_magl=self.cruise_magl,
            terrain=terrain,
            min_flight_path_angle=self.min_flight_path_angle,
            max_flight_path_angle=self.max_flight_path_angle,
            terminal_dive_angle=self.terminal_dive_angle,
            min_clearance=self.effective_min_clearance,
            sample_spacing=self.sample_spacing,
        )
        return GeneratedPath(
            times=[t_start + datetime.timedelta(seconds=float(t)) for t in path.times],
            lats=path.lats.tolist(),
            lons=path.lons.tolist(),
            alts=path.alts.tolist(),
        )


def build_trajectory(alpha: float, p_start: Point, p_stop: Point, dt: int = 1):
    """
    Calculate a ballistic trajectory that connects the start and stop point.

    The only force considered is gravity.

    Parameters
    ----------
    alpha: float
        Start angle of the ballistic motion [rad] in the 2D coordinate system
        (see "Returns")
    p_start: Point
        Start point of the trajectory
    p_stop: Point
        Stop point of the trajectory
    dt: int
        Number of seconds to use for representing the trajectory

    Returns
    -------
    v0: float
        Initial velocity [m/s]
    times: np.ndarray
        Time steps of the trajectory [s]. Shape (N,).
    trajectory: np.ndarray
        Points along the trajectory at 10s resolution. Shape (N, 2).
        Column 0 contains the horizontal component, column 1 the vertical component
        of the trajectory.
        The coordinate system is spanned by the LOS from start to end point and
        the parto of radial direction at the start point that is orthogonal to the LOS.
    """
    distance = line_of_sight_distance(*p_start.as_tuple(), *p_stop.as_tuple())

    p_start_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_start.as_tuple())
    )
    p_stop_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_stop.as_tuple())
    )
    direction = p_stop_ecef - p_start_ecef
    direction /= np.linalg.norm(direction)

    v0 = np.sqrt(9.81 * distance / (2 * np.cos(alpha) * np.sin(alpha)))
    t_stop = 2 * np.sin(alpha) * v0 / 9.81

    times = np.arange(0, t_stop, dt).tolist() + [t_stop]
    xs = []
    ys = []
    for t in times:
        x = np.cos(alpha) * v0 * t
        y = np.sin(alpha) * v0 * t - 0.5 * 9.81 * t**2
        xs.append(x)
        ys.append(y)

    if t_stop % 1 != 0:
        times.append(np.ceil(t_stop))
        xs.append(xs[-1])
        ys.append(ys[-1])

    return v0, times, np.vstack([xs, ys]).T


def convert_to_geodetic(
    p_start: Point,
    p_stop: Point,
    trajectory: np.ndarray,
) -> list[Point]:
    p_start_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_start.as_tuple())
    )
    p_stop_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_stop.as_tuple())
    )

    x_direction = p_stop_ecef - p_start_ecef
    x_direction /= np.linalg.norm(x_direction)
    y_direction = p_start_ecef / np.linalg.norm(p_start_ecef)
    y_direction -= np.dot(x_direction, y_direction) * x_direction
    assert np.isclose(np.dot(x_direction, y_direction), 0)

    points: list[Point] = []
    for x, y in trajectory:
        p_ecef = p_start_ecef + x * x_direction + y * y_direction
        p = CoordinateTransformations.cartesian_to_geodetic(*p_ecef)
        points.append(Point(lat=p[0], lon=p[1], alt=p[2]))
    return points



PathGenerator = Annotated[
    ClimbCruiseDiveGenerator | BallisticGenerator | TerrainFollowingGenerator,
    pydantic.Field(discriminator="type"),
]
