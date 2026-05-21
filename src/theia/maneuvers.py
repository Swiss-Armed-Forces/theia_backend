import abc
import datetime
import itertools

import numpy as np
import pydantic
from scipy.interpolate import CubicSpline

from theia.coordinates import (
    CoordinateTransformations,
    calculate_azimuth_angle,
    calculate_elevation_angle,
)
from theia.distance import line_of_sight_distance
from theia.measurement import MonostaticMeasurementTransformations
from theia.types import ConstantRcsModel, Point, Trajectory


class AbstractManeuver(abc.ABC):
    @abc.abstractmethod
    def get_waypoints(
        self,
        start_time: datetime.datetime,
        start_pos: Point,
    ) -> tuple[list[datetime.datetime], list[Point]]:
        raise NotImplementedError()

    def to_trajectory(
        self,
        start_time: datetime.datetime,
        start_pos: Point,
        target_id: int,
        sidc: str,
        rcs: float,
    ) -> Trajectory:
        times, points = self.get_waypoints(start_time, start_pos)

        timestamps = [t.timestamp() for t in times]
        points_ecef = np.array(
            [
                CoordinateTransformations.geodetic_to_cartesian(*p.as_tuple())
                for p in points
            ]
        )
        f_ecef = CubicSpline(timestamps, points_ecef, extrapolate=True)
        v_ecef = f_ecef.derivative()
        velocities = v_ecef(timestamps)

        return Trajectory(
            target_id=target_id,
            target_sidc=sidc,
            times=times,
            lats=[p.lat for p in points],
            lons=[p.lon for p in points],
            alts=[p.alt for p in points],
            vxs=[v[0] for v in velocities],
            vys=[v[1] for v in velocities],
            vzs=[v[2] for v in velocities],
            cross_section_model=ConstantRcsModel(rcs=rcs),
        )


class CompositeManeuver(AbstractManeuver):
    def __init__(self, maneuvers: list[AbstractManeuver]):
        self._maneuvers = maneuvers

    def get_waypoints(self, start_time, start_pos):
        all_times: list[datetime.datetime] = []
        all_points: list[Point] = []
        for i, maneuver in enumerate(self._maneuvers):
            times, points = maneuver.get_waypoints(start_time, start_pos)
            if i > 0:
                # Do not double count points.
                times = times[1:]
                points = points[1:]
            all_times.extend(times)
            all_points.extend(points)
            start_time = times[-1]
            start_pos = points[-1]
        return all_times, all_points


class ConstantSpeedStraightManeuver(AbstractManeuver, pydantic.BaseModel):
    stop_point: Point
    speed: float

    def get_waypoints(
        self,
        start_time: datetime.datetime,
        start_pos: Point,
    ) -> tuple[list[datetime.datetime], list[Point]]:
        res = 300

        d = line_of_sight_distance(
            start_pos.lat,
            start_pos.lon,
            start_pos.alt,
            self.stop_point.lat,
            self.stop_point.lon,
            self.stop_point.alt,
        )

        N = int(np.ceil(d / res))


        lats = np.linspace(start_pos.lat, self.stop_point.lat, N)
        lons = np.linspace(start_pos.lon, self.stop_point.lon, N)
        alts = np.linspace(start_pos.alt, self.stop_point.alt, N)

        lat_prev = lats[0]
        lon_prev = lons[0]
        alt_prev = alts[0]
        t_prev = start_time
        t = None
        points: list[Point] = []
        times: list[datetime.datetime] = []
        for lat, lon, alt in zip(lats[1:], lons[1:], alts[1:], strict=True):
            d = line_of_sight_distance(
                lat_prev,
                lon_prev,
                alt_prev,
                lat,
                lon,
                alt,
            )
            seconds = d / self.speed
            t = t_prev + datetime.timedelta(seconds=seconds)
            points.append(Point(lat=lat, lon=lon, alt=alt))
            times.append(t)

        return (
            times,
            points,
        )


class ConstantSpeedCurveManeuver(AbstractManeuver, pydantic.BaseModel):
    """Fly a curve at constant speed around a center point"""

    center_point: Point
    """Center point of the curve."""
    speed: float
    """Speed of flying the curve [m / s]"""
    angular_arclength: float
    """Angular length of the curve (clockwise, i. e. increasing azimuth) [rad]"""
    angular_res: float = 2 * np.pi / 360
    """Resolution of sampling the curve [rad]"""
    constant_altitude: bool = True
    """Whether to keep altitude constant (default) or fly at constant elevation to the center point"""

    def get_waypoints(
        self,
        start_time: datetime.datetime,
        start_pos: Point,
    ) -> tuple[list[datetime.datetime], list[Point]]:
        alt = start_pos.alt
        r = line_of_sight_distance(
            start_pos.lat,
            start_pos.lon,
            start_pos.alt,
            self.center_point.lat,
            self.center_point.lon,
            self.center_point.alt,
        )
        elevation = calculate_elevation_angle(self.center_point, start_pos)
        azimuth = calculate_azimuth_angle(self.center_point, start_pos)
        # if self.angular_arclength >= 0:
        #     azi1 = azimuth
        #     azi2 = azimuth + self.angular_arclength
        # else:
        #     azi1 = azimuth + self.angular_arclength
        #     azi2 = azimuth
        # azimuths = np.arange(
        #     azi1,
        #     azi2,
        #     self.angular_res,
        # )
        azimuths = np.arange(
            azimuth,
            azimuth + self.angular_arclength,
            np.sign(self.angular_arclength) * self.angular_res,
        )
        points: list[Point] = []
        for azimuth in azimuths:
            x, y, z = (
                MonostaticMeasurementTransformations.elevation_azimuth_range_to_cartesian(
                    self.center_point,
                    elevation,
                    azimuth,
                    r,
                )
            )
            lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
            points.append(
                Point(
                    lat=lat,
                    lon=lon,
                    alt=start_pos.alt if self.constant_altitude else alt,
                )
            )

        times: list[datetime.datetime] = [start_time]
        for p, p_next in itertools.pairwise(points):
            d = line_of_sight_distance(
                p.lat,
                p.lon,
                p.alt,
                p_next.lat,
                p_next.lon,
                p_next.alt,
            )
            dt = d / self.speed
            times.append(times[-1] + datetime.timedelta(seconds=dt))
        return times, points


class EightManeuver(AbstractManeuver, pydantic.BaseModel):
    center_point1: Point
    """First center point of the curve."""
    azimuth_center1_to_center2: float
    """Azimuth between the two center points [rad]"""
    elevation_center1_to_center2: float
    """Elevation between the two center points [rad]"""
    speed: float
    """Speed of flying the curve [m / s]"""
    angular_res: float = 2 * np.pi / 360
    """Resolution of sampling the curve [rad]"""

    def get_waypoints(self, start_time: datetime.datetime, start_pos: Point):
        center1 = self.center_point1
        r = line_of_sight_distance(
            start_pos.lat,
            start_pos.lon,
            start_pos.alt,
            center1.lat,
            center1.lon,
            center1.alt,
        )
        center2_ecef = (
            MonostaticMeasurementTransformations.elevation_azimuth_range_to_cartesian(
                center1,
                self.elevation_center1_to_center2,
                self.azimuth_center1_to_center2,
                2 * r,
            )
        )
        lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(*center2_ecef)
        center2 = Point(lat=lat, lon=lon, alt=alt)

        azimuth_start = calculate_azimuth_angle(self.center_point1, start_pos)
        d_azi = (self.azimuth_center1_to_center2 - azimuth_start + 2 * np.pi) % (
            2 * np.pi
        )

        maneuver = CompositeManeuver(
            [
                ConstantSpeedCurveManeuver(
                    center_point=center1,
                    speed=self.speed,
                    angular_arclength=d_azi,
                    angular_res=self.angular_res,
                ),
                ConstantSpeedCurveManeuver(
                    center_point=center2,
                    speed=self.speed,
                    angular_arclength=-np.sign(d_azi) * 2 * np.pi,
                    angular_res=self.angular_res,
                ),
                ConstantSpeedCurveManeuver(
                    center_point=center1,
                    speed=self.speed,
                    angular_arclength=np.sign(d_azi) * 2 * np.pi,
                    angular_res=self.angular_res,
                ),
            ]
        )
        return maneuver.get_waypoints(start_time, start_pos)
