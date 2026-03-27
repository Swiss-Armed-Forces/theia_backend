import numba
import numpy as np

from theia.coordinates import (
    CoordinateTransformations,
    calculate_azimuth_angle,
    calculate_elevation_angle,
)
from theia.distance import line_of_sight_distance
from theia.types import Point


class MonostaticMeasurementTransformations:
    @staticmethod
    @numba.njit
    def elevation_azimuth_range_to_enu(
        elevation: float, azimuth: float, range_m: float
    ) -> tuple[float, float, float]:
        east = range_m * np.cos(elevation) * np.sin(azimuth)
        north = range_m * np.cos(elevation) * np.cos(azimuth)
        up = range_m * np.sin(elevation)

        return east, north, up

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

        return CoordinateTransformations.enu_to_ecef(
            observer_point,
            MonostaticMeasurementTransformations.elevation_azimuth_range_to_enu(
                elevation,
                azimuth,
                range_m,
            ),
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
        east, north, up = CoordinateTransformations.ecef_to_enu(
            observer_point,
            p,
        )

        return MonostaticMeasurementTransformations.enu_to_elevation_azimuth_range(
            east,
            north,
            up,
        )

    @staticmethod
    @numba.njit
    def enu_to_elevation_azimuth_range(
        east: float, north: float, up: float
    ) -> tuple[float, float, float]:
        range_m = np.sqrt(east**2 + north**2 + up**2)
        horizontal_range = np.sqrt(east**2 + north**2)
        elevation = np.arctan2(up, horizontal_range)
        azimuth = np.arctan2(east, north)

        return elevation, azimuth, range_m
