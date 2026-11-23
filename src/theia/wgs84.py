"""
WGS84 ellipsoid: constants and numba kernels for coordinate conversions.

This module must not import from other theia modules, so that low-level
modules such as `theia.types` can use it without circular imports. Use
`theia.coordinates.CoordinateTransformations` for the regular API.
"""

import math

import numba

# WGS84 ellipsoid parameters.
WGS84_A = 6378137.0
"""Semi-major axis [m]"""
WGS84_F = 1.0 / 298.257223563
"""Flattening"""
WGS84_B = WGS84_A * (1.0 - WGS84_F)
"""Semi-minor axis [m]"""
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
"""First eccentricity squared"""


@numba.njit
def geodetic_to_cartesian_impl(
    lat: float, lon: float, alt: float
) -> tuple[float, float, float]:
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # Radius of curvature in the prime vertical.
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (N + alt) * cos_lat * math.cos(lon_rad)
    y = (N + alt) * cos_lat * math.sin(lon_rad)
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat
    return x, y, z


@numba.njit
def cartesian_to_geodetic_impl(
    x: float, y: float, z: float
) -> tuple[float, float, float]:
    p = math.sqrt(x * x + y * y)

    if p < 1e-9:
        # On (or numerically indistinguishable from) the polar axis:
        # longitude is undefined and latitude is exactly +/-90 deg, so the
        # general iterative solution below (which divides by cos(lat) and
        # p) would blow up. Altitude reduces to a 1-D expression along the
        # semi-minor axis.
        lat = math.pi / 2.0 if z >= 0.0 else -math.pi / 2.0
        alt = abs(z) - WGS84_B
        return math.degrees(lat), 0.0, alt

    lon = math.atan2(y, x)

    # Bowring's iterative method (1976): converges to sub-nanometer altitude
    # accuracy within a handful of Newton-Raphson-like iterations on the
    # geodetic latitude, for any latitude/altitude. A fixed iteration count
    # is used (rather than a convergence-check loop) since it's cheap
    # relative to the trig calls and keeps the function branch-free after
    # the pole check above. 4 iterations is the minimum that keeps a
    # comfortable margin against
    # tests/test_coordinate_transformations_pyproj_reference.py (verified: 4
    # passes with ~2-5x margin on every tolerance there, 3 fails).
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(4):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * N / (N + alt)))

    # Recompute alt against the final converged lat (the alt computed inside
    # the last loop iteration used the pre-update lat's N).
    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), alt


