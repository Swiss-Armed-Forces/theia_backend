"""
Terrain-following cruise missile trajectories.

The missile flies along the great circle from the start to the target point at a
constant speed. Vertically, it follows the terrain at a nominal height above
ground level (AGL) while respecting flight-path angle limits, and finishes with
a straight terminal dive onto the target.

The cruise profile is the lowest profile that never goes below the nominal AGL
height and whose flight-path angles stay within the configured limits. It climbs
early enough to clear upcoming ridges and stays high over narrow valleys. If the
terrain right after the launch rises faster than the missile can climb, the
missile is launched correspondingly higher.

All altitudes are metres above sea level, consistent with
:meth:`theia.terrain.AbstractTerrainModel.elevationAt`.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from theia.config import (
    CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE,
    CM_DEFAULT_MIN_CLEARANCE_FACTOR,
    CM_DEFAULT_MIN_CLEARANCE_FLOOR,
    CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE,
    CM_DEFAULT_SAMPLE_SPACING,
    CM_DEFAULT_TERMINAL_DIVE_ANGLE,
    CM_DIVE_PENETRATION_TOLERANCE,
    CM_IMPACT_HOLD_OFFSETS,
    CM_MIN_KNOT_SPACING_FACTOR,
    CM_PRE_IMPACT_OFFSETS,
    CM_TERRAIN_SAMPLE_SPACING,
)
from theia.distance import R_EARTH, haversine
from theia.terrain import AbstractTerrainModel
from theia.types import Point


def default_min_clearance(cruise_magl: float) -> float:
    """Default minimum clearance above the terrain [m]."""
    return min(
        cruise_magl,
        max(
            CM_DEFAULT_MIN_CLEARANCE_FLOOR,
            CM_DEFAULT_MIN_CLEARANCE_FACTOR * cruise_magl,
        ),
    )


def validate_parameters(
    p_start: Point,
    p_stop: Point,
    speed: float,
    cruise_magl: float,
    min_flight_path_angle: float,
    max_flight_path_angle: float,
    terminal_dive_angle: float,
    min_clearance: float,
    sample_spacing: float,
) -> None:
    """Raise ``ValueError`` if the parameters do not describe a valid flight."""
    if speed <= 0:
        raise ValueError(f"speed must be > 0 m/s, got {speed}")
    if cruise_magl <= 0:
        raise ValueError(f"cruise_magl must be > 0 m, got {cruise_magl}")
    if not -90 < min_flight_path_angle < 0:
        raise ValueError(
            f"min_flight_path_angle must lie in (-90°, 0°), got {min_flight_path_angle}"
        )
    if not 0 < max_flight_path_angle < 90:
        raise ValueError(
            f"max_flight_path_angle must lie in (0°, 90°), got {max_flight_path_angle}"
        )
    if not -90 < terminal_dive_angle < 0:
        raise ValueError(
            f"terminal_dive_angle must lie in (-90°, 0°), got {terminal_dive_angle}"
        )
    if not 0 < min_clearance <= cruise_magl:
        raise ValueError(
            f"min_clearance must lie in (0, cruise_magl={cruise_magl}] m, "
            f"got {min_clearance}"
        )
    if sample_spacing <= 0:
        raise ValueError(f"sample_spacing must be > 0 m, got {sample_spacing}")
    distance = haversine(p_start.lon, p_start.lat, p_stop.lon, p_stop.lat)
    if distance < 2 * sample_spacing:
        raise ValueError(
            f"Start and target are {distance:.0f} m apart; "
            f"at least 2 * sample_spacing = {2 * sample_spacing:.0f} m are required"
        )


@dataclass
class CruiseMissilePath:
    times: np.ndarray
    """Seconds since launch. Shape (N,)."""
    lats: np.ndarray
    """Latitudes [°]. Shape (N,)."""
    lons: np.ndarray
    """Longitudes [°]. Shape (N,)."""
    alts: np.ndarray
    """Altitudes [m above sea level]. Shape (N,)."""
    impact: Point
    """Point where the missile hits the ground at the target."""
    t_impact: float
    """Seconds since launch at which the missile hits the target."""


def build_terrain_following_path(
    p_start: Point,
    p_stop: Point,
    speed: float,
    cruise_magl: float,
    terrain: AbstractTerrainModel,
    min_flight_path_angle: float = CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE,
    max_flight_path_angle: float = CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE,
    terminal_dive_angle: float = CM_DEFAULT_TERMINAL_DIVE_ANGLE,
    min_clearance: float | None = None,
    sample_spacing: float = CM_DEFAULT_SAMPLE_SPACING,
) -> CruiseMissilePath:
    """
    Compute a terrain-following cruise missile path.

    The altitudes of ``p_start`` and ``p_stop`` are ignored: the missile starts
    at least ``cruise_magl`` above the terrain (higher if needed to clear the
    terrain right after the launch) and hits the terrain at the target.

    Parameters
    ----------
    p_start: Point
        Launch point.
    p_stop: Point
        Target point.
    speed: float
        Constant speed along the 3D path [m/s].
    cruise_magl: float
        Nominal cruise height above ground level [m].
    terrain: AbstractTerrainModel
        Terrain to follow.
    min_flight_path_angle: float
        Steepest descent while cruising [°], in (-90°, 0°).
    max_flight_path_angle: float
        Steepest climb while cruising [°], in (0°, 90°).
    terminal_dive_angle: float
        Flight-path angle of the terminal dive [°], in (-90°, 0°). Like the
        other flight-path angles, it is negative when descending.
    min_clearance: float | None
        Clearance above the terrain [m] that the interpolated trajectory must
        keep while cruising. ``None`` selects :func:`default_min_clearance`.
    sample_spacing: float
        Distance between trajectory samples along the ground track [m].

    Raises
    ------
    ValueError
        If the parameters are invalid or the terrain makes the flight impossible.
    """
    if min_clearance is None:
        min_clearance = default_min_clearance(cruise_magl)
    validate_parameters(
        p_start,
        p_stop,
        speed,
        cruise_magl,
        min_flight_path_angle,
        max_flight_path_angle,
        terminal_dive_angle,
        min_clearance,
        sample_spacing,
    )

    distance = haversine(p_start.lon, p_start.lat, p_stop.lon, p_stop.lat)

    # Terrain profile on a fine grid. Its spacing divides sample_spacing so
    # that every regular output sample lies on the grid.
    n_sub = int(math.ceil(sample_spacing / CM_TERRAIN_SAMPLE_SPACING))
    s_fine = np.arange(0.0, distance, sample_spacing / n_sub)
    s_fine = np.append(s_fine[s_fine < distance - 1e-6], distance)
    lats_fine, lons_fine = great_circle_points(p_start, p_stop, s_fine)
    h_fine = _elevations(terrain, lats_fine, lons_fine)

    # Cruise profile.
    desired = h_fine + cruise_magl
    # If the terrain right after the launch rises faster than the missile can
    # climb, the envelope starts higher than cruise_magl above the launch
    # point. This is intended: the missile is launched high enough to clear it.
    z_cruise = rate_limited_envelope(
        s_fine,
        desired,
        math.tan(math.radians(min_flight_path_angle)),
        math.tan(math.radians(max_flight_path_angle)),
    )

    # Terminal dive.
    # Intersect the dive line mandated by the dive angle with the cruise profile.
    h_target = float(h_fine[-1])
    tan_dive = math.tan(math.radians(terminal_dive_angle))

    def dive_line(s):
        # The dive descends (tan_dive < 0), so the line rises backwards from
        # the target.
        return h_target - tan_dive * (distance - s)

    s_dive = _dive_start(s_fine, z_cruise, dive_line(s_fine))
    in_dive = s_fine >= s_dive
    blocked = dive_line(s_fine[in_dive]) < h_fine[in_dive] - 1e-6
    if blocked.any():
        s_blocked = s_fine[in_dive][np.argmax(blocked)]
        raise ValueError(
            f"Terminal dive at {terminal_dive_angle}° is blocked by terrain "
            f"{distance - s_blocked:.0f} m before the target"
        )

    def profile(s):
        """Stitch the cruise line and the dive line together."""
        return np.where(s < s_dive, np.interp(s, s_fine, z_cruise), dive_line(s))

    # Output samples: regular spacing plus fixed samples at the dive start (a
    # kink), shortly before the impact on the dive line (see
    # CM_PRE_IMPACT_OFFSETS) and at the impact. Regular samples too close to
    # the fixed ones are dropped to avoid near-duplicate spline knots.
    pre_impact_ds = (
        speed
        * np.array(CM_PRE_IMPACT_OFFSETS)
        * math.cos(math.radians(terminal_dive_angle))
    )
    s_pre_impact = distance - pre_impact_ds
    # Only on the dive line, and not right next to the dive start.
    s_pre_impact = s_pre_impact[
        s_pre_impact - s_dive >= np.min(pre_impact_ds, initial=np.inf)
    ]
    s_fixed = np.concatenate([[s_dive], s_pre_impact, [distance]])
    s_regular = s_fine[:-1:n_sub]
    too_close = np.abs(s_regular[:, None] - s_fixed[None, :]).min(axis=1) < (
        CM_MIN_KNOT_SPACING_FACTOR * sample_spacing
    )
    too_close[0] = False
    s_out = np.union1d(s_regular[~too_close], s_fixed)

    lats_out, lons_out = great_circle_points(p_start, p_stop, s_out)
    alts_out = profile(s_out)

    # Time from the 3D path length between the samples, i.e. along the
    # straight segments the simulated (interpolated) trajectory follows.
    path_length = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(s_out), np.diff(alts_out)))]
    )
    t_out = path_length / speed

    # Duplicate the last point a couple of times to ensure the simulation timestep
    # does not miss the impact and avoid issues with spline interpolation.
    t_impact = float(t_out[-1])
    times = np.concatenate([t_out, t_impact + np.array(CM_IMPACT_HOLD_OFFSETS)])
    lats = np.concatenate(
        [lats_out, np.full(len(CM_IMPACT_HOLD_OFFSETS), lats_out[-1])]
    )
    lons = np.concatenate(
        [lons_out, np.full(len(CM_IMPACT_HOLD_OFFSETS), lons_out[-1])]
    )
    alts = np.concatenate(
        [alts_out, np.full(len(CM_IMPACT_HOLD_OFFSETS), alts_out[-1])]
    )

    _check_clearance(
        terrain,
        times,
        lats,
        lons,
        alts,
        t_eval=np.interp(s_fine, s_out, t_out),
        is_dive=in_dive,
        min_clearance=min_clearance,
    )

    return CruiseMissilePath(
        times=times,
        lats=lats,
        lons=lons,
        alts=alts,
        impact=Point(lat=lats_out[-1], lon=lons_out[-1], alt=alts_out[-1]),
        t_impact=t_impact,
    )


def great_circle_points(
    p_start: Point,
    p_stop: Point,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Points along the great circle from ``p_start`` to ``p_stop``.

    Uses the same spherical earth (``R_EARTH``) as
    :func:`theia.distance.haversine`.

    Parameters
    ----------
    s: np.ndarray
        Arc lengths from ``p_start`` [m].

    Returns
    -------
    lats, lons: np.ndarray
        Coordinates [°] of the points.
    """
    lat1, lon1 = math.radians(p_start.lat), math.radians(p_start.lon)
    lat2, lon2 = math.radians(p_stop.lat), math.radians(p_stop.lon)
    u1 = np.array(
        [
            math.cos(lat1) * math.cos(lon1),
            math.cos(lat1) * math.sin(lon1),
            math.sin(lat1),
        ]
    )
    u2 = np.array(
        [
            math.cos(lat2) * math.cos(lon2),
            math.cos(lat2) * math.sin(lon2),
            math.sin(lat2),
        ]
    )
    # Spherical linear interpolation (SLERP, Shoemake 1985 https://doi.org/10.1145/325165.325242)
    # between the unit vectors u1 and u2, which enclose the central angle omega
    # (u1 · u2 = cos(omega)).
    #
    # Derivation: e1 = u1 and e2 = (u2 - cos(omega) u1) / sin(omega)
    # (Gram-Schmidt) are orthonormal and span the plane of the great circle.
    # The unit vector at angle f from u1 towards u2 is therefore
    #
    #   p(f) = cos(f) e1 + sin(f) e2
    #        = [cos(f) sin(omega) - sin(f) cos(omega)] / sin(omega) u1
    #          + sin(f) / sin(omega) u2
    #        = sin(omega - f) / sin(omega) u1 + sin(f) / sin(omega) u2.
    #
    # p(f) has unit length and p(f) · u1 = cos(f), so the point lies exactly
    # R_EARTH * f along the great circle. Hence f = s / R_EARTH. Normalising a
    # straight-line blend of u1 and u2 would stay on the great circle too, but
    # would not space the points evenly. Undefined for omega = 0 (identical
    # points) and omega = pi (antipodal points).
    omega = haversine(p_start.lon, p_start.lat, p_stop.lon, p_stop.lat) / R_EARTH
    f = np.asarray(s, dtype=float) / R_EARTH
    a = np.sin(omega - f) / math.sin(omega)
    b = np.sin(f) / math.sin(omega)
    xyz = a[:, None] * u1 + b[:, None] * u2
    lats = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    lons = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return lats, lons


def rate_limited_envelope(
    s: np.ndarray,
    desired: np.ndarray,
    tan_min: float,
    tan_max: float,
) -> np.ndarray:
    """
    Lowest profile ``z >= desired`` whose slopes lie in ``[tan_min, tan_max]``.

    The backward pass starts climbs early enough to clear upcoming terrain; the
    forward pass limits the descent rate. The forward pass only raises points,
    which keeps the climb limit satisfied.

    Parameters
    ----------
    s: np.ndarray
        Increasing distances along the track [m].
    desired: np.ndarray
        Minimum altitude at each distance [m].
    tan_min: float
        Steepest allowed descent as slope (negative).
    tan_max: float
        Steepest allowed climb as slope (positive).
    """
    z = np.array(desired, dtype=float)
    ds = np.diff(s)
    for i in range(len(z) - 2, -1, -1):
        z[i] = max(z[i], z[i + 1] - tan_max * ds[i])
    for i in range(1, len(z)):
        z[i] = max(z[i], z[i - 1] + tan_min * ds[i - 1])
    return z


def _dive_start(s: np.ndarray, z_cruise: np.ndarray, z_dive: np.ndarray) -> float:
    """Distance at which the dive line last crosses the cruise profile."""
    above = z_dive >= z_cruise
    if not above[0]:
        raise ValueError(
            "Target is too close: the terminal dive would start before the launch"
        )
    i = len(above) - 1 - int(np.argmax(above[::-1]))
    if i == len(above) - 1:
        return float(s[-1])
    # Linear interpolation of the crossing between s[i] and s[i + 1].
    d0 = z_dive[i] - z_cruise[i]
    d1 = z_dive[i + 1] - z_cruise[i + 1]
    return float(s[i] + (s[i + 1] - s[i]) * d0 / (d0 - d1))


def _elevations(
    terrain: AbstractTerrainModel,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    return np.array(
        [terrain.elevationAt(float(lat), float(lon)) for lat, lon in zip(lats, lons)]
    )


def _check_clearance(
    terrain: AbstractTerrainModel,
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    alts: np.ndarray,
    t_eval: np.ndarray,
    is_dive: np.ndarray,
    min_clearance: float,
) -> None:
    """
    Check the trajectory as interpolated by :class:`theia.types.Trajectory`.

    While cruising, the interpolated altitude must stay ``min_clearance`` above
    the terrain. During the dive, it must not go below the terrain.
    """
    spline = CubicSpline(times, np.stack([lats, lons, alts], axis=1))
    lat_i, lon_i, alt_i = spline(t_eval).T
    clearance = alt_i - _elevations(terrain, lat_i, lon_i)
    required = np.where(is_dive, -CM_DIVE_PENETRATION_TOLERANCE, min_clearance)
    violation = required - clearance
    worst = int(np.argmax(violation))
    if violation[worst] > 1e-6:
        phase = "terminal dive" if is_dive[worst] else "cruise"
        raise ValueError(
            f"Interpolated trajectory comes within {clearance[worst]:.1f} m of "
            f"the terrain during the {phase} (required: {required[worst]:.1f} m) "
            f"at t = {t_eval[worst]:.1f} s; reduce sample_spacing"
        )
