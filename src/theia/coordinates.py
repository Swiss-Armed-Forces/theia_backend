import math

import numba
import numpy as np

from theia.types import Point, Velocity

LATLON_BOUNDS = {
    "CH": {"lat": [45.7, 45.9], "lon": [5.7, 10.6]},
}

POSITIONS_OF_INTEREST = {
    "CH_CENTER": {"lat": 46.801111, "lon": 8.226667},
    "Uetliberg": {"lat": 47.349491, "lon": 8.492063, "alt": 856.2037851199802},
}

# WGS84 ellipsoid parameters.
_WGS84_A = 6378137.0
"""Semi-major axis [m]"""
_WGS84_F = 1.0 / 298.257223563
"""Flattening"""
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
"""Semi-minor axis [m]"""
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
"""First eccentricity squared"""


@numba.njit
def _geodetic_to_cartesian_impl(
    lat: float, lon: float, alt: float
) -> tuple[float, float, float]:
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # Radius of curvature in the prime vertical.
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)

    x = (N + alt) * cos_lat * math.cos(lon_rad)
    y = (N + alt) * cos_lat * math.sin(lon_rad)
    z = (N * (1.0 - _WGS84_E2) + alt) * sin_lat
    return x, y, z


@numba.njit
def _cartesian_to_geodetic_impl(
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
        alt = abs(z) - _WGS84_B
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
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(4):
        sin_lat = math.sin(lat)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1.0 - _WGS84_E2 * N / (N + alt)))

    # Recompute alt against the final converged lat (the alt computed inside
    # the last loop iteration used the pre-update lat's N).
    sin_lat = math.sin(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), alt


class CoordinateTransformations:
    @classmethod
    def geodetic_to_cartesian(
        cls, lat: float, lon: float, alt: float
    ) -> tuple[float, float, float]:
        x, y, z = _geodetic_to_cartesian_impl(lat, lon, alt)
        return float(x), float(y), float(z)

    @classmethod
    def cartesian_to_geodetic(
        cls,
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float, float]:
        """Convert (x, y, z) to (lat, lon, alt)."""
        lat, lon, alt = _cartesian_to_geodetic_impl(x, y, z)
        return float(lat), float(lon), float(alt)

    @staticmethod
    def velocity_cartesian_to_geodetic(
        p: Point,
        velocity: Velocity,
    ) -> tuple[float, float, float]:
        """Project velocity to latitude, longitude and altitude directions at point p [m / s]"""
        n_lat = latitude_direction(p)
        n_lon = longitude_direction(p)
        n_alt = altitude_direction(p)

        v = velocity.as_tuple()

        return (
            np.dot(n_lat, v),
            np.dot(n_lon, v),
            np.dot(n_alt, v),
        )

    @staticmethod
    def velocity_geodetic_to_cartesian(
        p: Point,
        vlat: float,
        vlon: float,
        valt: float,
    ) -> Velocity:
        n_lat = latitude_direction(p)
        n_lon = longitude_direction(p)
        n_alt = altitude_direction(p)

        v_cartesian = vlat * n_lat + vlon * n_lon + valt * n_alt

        return Velocity(
            vx=v_cartesian[0],
            vy=v_cartesian[1],
            vz=v_cartesian[2],
        )


class EcefToEnuTransformer:
    def __init__(self, reference_point: Point):
        """
        Parameters
        ----------
        reference_point: Point
            Reference point in geodetic coordinates at which the
            ENU-frame is defined. Typically the observer (radar) position.
        """
        self._R = _ecef_to_enu_rotation_matrix(
            reference_point.lat,
            reference_point.lon,
        )
        self._RT = np.array(self._R.T)
        reference_point_xyz = np.array(
            CoordinateTransformations.geodetic_to_cartesian(
                *reference_point.as_tuple(),
            )
        )
        self._center = np.array(reference_point_xyz)

    def ecef_to_enu(
        self,
        p_ecef: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """
        Convert Cartesian coordinates from earth-centered-earth-fixed to east-north-up.

        Parameters
        ----------
        p_ecef: tuple[float, float, float]
            Point in Cartesian ECEF coordinates to be transformed to ENU
            coordinates. Typically the observed (target) position.

        Returns
        -------
        tuple[float, float, float]
            Cartesian ENU coordinates corresponding to p_ecef

        Notes
        -----
        Formula according to Wikipedia:
        https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU
        """
        p_ecef = np.array(p_ecef)
        p_enu = self._R @ (p_ecef - self._center)
        return (float(p_enu[0]), float(p_enu[1]), float(p_enu[2]))

    def ecef_to_enu_multiple(
        self,
        points_ecef: list[tuple[float, float, float]],
    ) -> np.ndarray:
        """
        Convert Cartesian coordinates from earth-centered-earth-fixed to east-north-up.

        Parameters
        ----------
        p_ecef: tuple[float, float, float]
            Point in Cartesian ECEF coordinates to be transformed to ENU
            coordinates. Typically the observed (target) position.

        Returns
        -------
        tuple[float, float, float]
            Cartesian ENU coordinates corresponding to p_ecef

        Notes
        -----
        Formula according to Wikipedia:
        https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU
        """
        points = np.array(points_ecef)
        p_enu = (points - self._center) @ self._RT
        return p_enu

    def enu_to_ecef(
        self,
        p_enu: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """
        Convert Cartesian coordinates from east-north-up to earth-centered-earth-fixed.

        Parameters
        ----------
        p_enu: tuple[float, float, float]
            Point in Cartesian ENU coordinates to be transformed to ECEF
            coordinates. Typically the observed (target) position.

        Returns
        -------
        tuple[float, float, float]
            Cartesian ECEF coordinates corresponding to p_enu

        Notes
        -----
        Formula according to Wikipedia:
        https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF
        """
        R_enu_to_ecef = self._RT

        reference_point_xyz = np.array(self._center)
        p_enu = np.array(p_enu)
        p_ecef = R_enu_to_ecef @ p_enu + reference_point_xyz
        return (float(p_ecef[0]), float(p_ecef[1]), float(p_ecef[2]))


@numba.njit
def _ecef_to_enu_rotation_matrix(lat: float, lon: float) -> np.array:
    """
    Calculate rotation matrix that converts earth-centered-earth-fixed to East-North-Up
    coordinates at the given lat, lon coordinates.

    Parameters
    ----------
    lat: float
        Latitude [°]
    lon: float
        Longitude [°]

    Returns
    -------
    np.ndarray
        Rotation matrix of shape (3, 3)
    """
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    # fmt: off
    R_ecef_to_enu = np.empty(shape=(3, 3), dtype=np.float64)
    R_ecef_to_enu[0, :] = (-np.sin(lon)              ,  np.cos(lon)              ,         0.0)
    R_ecef_to_enu[1, :] = (-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat))
    R_ecef_to_enu[2, :] = ( np.cos(lat) * np.cos(lon),  np.cos(lat) * np.sin(lon), np.sin(lat))
    # fmt: on
    return R_ecef_to_enu


# The following function is taken from openBURST.
def calculate_azimuth_angle(p_observer: Point, p_target: Point) -> float:
    """
    Calculate azimuth angle between an observer and a target [rad].

    The azimuth is defined as the clockwise angle from north between the observer
    and the target, i. e. the angle that goes from the observer towards the target.

    Parameters
    ----------
    p_observer: Point
        Position of the observer
    p_target: Point
        Position of the target

    Returns
    -------
    float
        azimuth [rad] in the interval (0, 2pi)

    References
    ----------
    https://geographiclib.sourceforge.io/2009-03/geodesic.html

    """
    tgt_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_target.as_tuple())
    )
    transformer = EcefToEnuTransformer(p_observer)
    east, north, up = transformer.ecef_to_enu(tgt_ecef)
    return np.arctan2(east, north) % (2 * np.pi)


# The following function was generated using Claude AI Sonnet 4.5
# and adapted by the author.
@numba.njit
def calculate_elevation_angle_ecef(
    p_observer: np.array,
    p_target: np.array,
) -> float:
    delta = p_target - p_observer

    # Local "up" vector at p1 (radial direction from Earth's center).
    # This is simply the normalized position vector of p1.
    up = p_observer / np.linalg.norm(p_observer)

    # The elevation angle is 90° minus the angle between los and "up"
    # Or equivalently: arcsin(dot_product / los_magnitude)
    elevation_angle_rad = np.asin(np.dot(up, delta) / np.linalg.norm(delta))

    return elevation_angle_rad


def calculate_elevation_angle(p_observer: Point, p_target: Point):
    """
    Calculate the elevation angle from observer to target [rad].

    This takes earth's curvature into account.

    Parameters
    ----------
    p_observer: Point
        Position of the observer.
    p_target: Point
        Position of the target.

    Returns
    -------
    elevation_angle: float
        elevation angle [rad] in the interval [-pi/2, pi/2]
        Positive elevation means that the target is above the observer's horizon.
        Negative elevation means that the target is below the observer's horizon.
    """
    p1_xyz = np.asarray(
        CoordinateTransformations.geodetic_to_cartesian(*p_observer.as_tuple())
    )
    p2_xyz = np.asarray(
        CoordinateTransformations.geodetic_to_cartesian(*p_target.as_tuple())
    )

    return calculate_elevation_angle_ecef(p1_xyz, p2_xyz)


def latitude_direction(p: Point) -> np.ndarray:
    """
    Calculate the direction in Cartesian coordinates of the latitude axis at geodetic point p.

    North direction: tangent to meridian, pointing toward increasing latitude.
    """
    lat_rad = np.radians(p.lat)
    lon_rad = np.radians(p.lon)

    return np.array(
        [
            -np.sin(lat_rad) * np.cos(lon_rad),
            -np.sin(lat_rad) * np.sin(lon_rad),
            np.cos(lat_rad),
        ]
    )


def longitude_direction(p: Point) -> np.ndarray:
    """
    Calculate the direction in Cartesian coordinates of the longitude axis at geodetic point p.

    East direction: tangent to parallel, pointing toward increasing longitude.
    """
    lon_rad = np.radians(p.lon)

    return np.array(
        [
            -np.sin(lon_rad),
            np.cos(lon_rad),
            0.0,
        ]
    )


def altitude_direction(p: Point) -> np.ndarray:
    """
    Calculate the direction in Cartesian coordinates of the altitude axis at geodetic point p.

    Up direction: normal to WGS84 ellipsoid surface.
    """
    lat_rad = np.radians(p.lat)
    lon_rad = np.radians(p.lon)

    # For WGS84 ellipsoid, the normal direction is:
    # (Note: this differs from radial direction due to Earth's flattening)
    return np.array(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )
