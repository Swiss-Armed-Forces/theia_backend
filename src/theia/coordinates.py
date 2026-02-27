import numpy as np
import pyproj
from geographiclib.geodesic import Geodesic
from theia.terrain import elevationAt
from theia.types import Point, Velocity


LATLON_BOUNDS = {
    "CH": {"lat": [45.7, 45.9], "lon": [5.7, 10.6]},
}

POSITIONS_OF_INTEREST = {
    "CH_CENTER": {"lat": 46.801111, "lon": 8.226667},
}


def sample_location(
    rng: np.random.Generator,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Point:
    lat = rng.uniform(lat_min, lat_max)
    lon = rng.uniform(lon_min, lon_max)
    alt = elevationAt(lat, lon)
    return Point(lat=lat, lon=lon, alt=alt)


class CoordinateTransformations:
    t = pyproj.Transformer.from_proj(
        pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84"),
        pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84"),
    )

    @classmethod
    def geodetic_to_cartesian(
        cls, lat: float, lon: float, alt: float
    ) -> tuple[float, float, float]:
        # the parameter sequence should be lon, lat, alt!
        x, y, z = cls.t.transform(lon, lat, alt, radians=False)
        return float(x), float(y), float(z)

    @classmethod
    def cartesian_to_geodetic(
        cls,
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float, float]:
        """Convert (x, y, z) to (lat, lon, alt)."""
        lon, lat, alt = cls.t.transform(x, y, z, radians=False, direction="INVERSE")
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

    @staticmethod
    def ecef_to_enu(
        reference_point: Point, p_ecef: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """
        Convert Cartesian coordinates from earth-centered-earth-fixed to east-north-up.

        Parameters
        ----------
        reference_point: Point
            Reference point in geodetic coordinates at which the
            ENU-frame is defined. Typically the observer (radar) position.
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
        R_ecef_to_enu = _ecef_to_enu_rotation_matrix(
            reference_point.lat,
            reference_point.lon,
        )

        reference_point_xyz = np.array(
            CoordinateTransformations.geodetic_to_cartesian(
                *reference_point.as_tuple(),
            )
        )
        p_ecef = np.array(p_ecef)
        return tuple(R_ecef_to_enu @ (p_ecef - reference_point_xyz))

    @staticmethod
    def enu_to_ecef(
        reference_point: Point, p_enu: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """
        Convert Cartesian coordinates from east-north-up to earth-centered-earth-fixed.

        Parameters
        ----------
        reference_point: Point
            Reference point in geodetic coordinates at which the
            ENU-frame is defined. Typically the observer (radar) position.
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
        R_ecef_to_enu = _ecef_to_enu_rotation_matrix(
            reference_point.lat,
            reference_point.lon,
        )
        R_enu_to_ecef = R_ecef_to_enu.T

        reference_point_xyz = np.array(
            CoordinateTransformations.geodetic_to_cartesian(
                *reference_point.as_tuple(),
            )
        )
        p_enu = np.array(p_enu)
        return tuple(R_enu_to_ecef @ p_enu + reference_point_xyz)


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
    R_ecef_to_enu = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )
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
    # Inverse() returns angles in (-180, 180), but we need (0, 360).
    tmp = Geodesic.WGS84.Inverse(
        p_observer.lat,
        p_observer.lon,
        p_target.lat,
        p_target.lon,
    )["azi1"]
    tmp = (tmp + 360) % 360
    return np.radians(tmp)


# The following function was generated using Claude AI Sonnet 4.5
# and adapted by the author.
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

    delta = p2_xyz - p1_xyz

    # Local "up" vector at p1 (radial direction from Earth's center).
    # This is simply the normalized position vector of p1.
    up = p1_xyz / np.linalg.norm(p1_xyz)

    # The elevation angle is 90° minus the angle between los and "up"
    # Or equivalently: arcsin(dot_product / los_magnitude)
    elevation_angle_rad = np.asin(np.dot(up, delta) / np.linalg.norm(delta))

    return elevation_angle_rad


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
