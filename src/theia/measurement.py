import numpy as np

from theia.coordinates import CoordinateTransformations, calculate_azimuth_angle, calculate_elevation_angle
from theia.distance import line_of_sight_distance
from theia.types import Point


class MonostaticMeasurementTransformations():

    @staticmethod
    def elevation_azimuth_range_to_cartesian(
        observer_point: Point,
        elevation: float,
        azimuth: float,
        range_m: float,
    ) -> tuple[float, float, float]:
        """
        Convert (elevation, azimuth, range) coordinates defined at the reference
        point's ENU-frame to Cartesian ECEF coordinates.

        Parameters
        ----------
        observer_point: Point
            Position of the observer
        elevation: float
            Elevation angle (upward from horizontal plane) [rad]
        azimuth: float
            Azimuth angle (clockwise from North) [rad]
        range_m: float
            Distance between observer and target [m]

        Returns
        -------
        tuple[float, float, float]
            Observed point in Cartesian ECEF coordinates [m]
        """
        east = range_m * np.cos(elevation) * np.sin(azimuth)
        north = range_m * np.cos(elevation) * np.cos(azimuth)
        up = range_m * np.sin(elevation)

        return CoordinateTransformations.enu_to_ecef(
            observer_point,
            (east, north, up),
        )

    @staticmethod
    def cartesian_to_elevation_azimuth_range(
        observer_point: Point,
        p: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """
        Calculate (elevation, azimuth, range) as measured by an observer at observer_point.

        Parameters
        ----------
        observer_point: Point
            Position of the observer
        p: tuple[float, float, float]
            Position in Cartesian ECEF coordinates to measure

        Return
        ------
        tuple[float, float, float]
            elevation angle [rad], azimuth angle [rad] and range
            (i. e. line-of-sight distance to the observer) [m]
        """
        lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(*p)
        p_target = Point(lat=lat, lon=lon, alt=alt)
        elevation = calculate_elevation_angle(observer_point, p_target)
        azimuth = calculate_azimuth_angle(observer_point, p_target)
        range = line_of_sight_distance(
            observer_point.lat,
            observer_point.lon,
            observer_point.alt,
            lat,
            lon,
            alt,
        )
        return elevation, azimuth, range