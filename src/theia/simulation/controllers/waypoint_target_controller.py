from __future__ import annotations
import datetime

import numpy as np
from scipy.interpolate import CubicSpline
from theia.coordinates import CoordinateTransformations
from theia.types import (
    Controller,
    Point,
    Sensor,
    RcsModel,
    Receiver,
    SituationalPicture,
    Target,
    Trajectory,
    Transmitter,
    Velocity,
)


class WaypointTargetController(Controller):
    def __init__(
        self,
        target_id: int,
        times: list[datetime.datetime],
        waypoints: list[Point],
        rcs_model: RcsModel,
    ):
        assert len(times) == len(waypoints)
        self._target_id = target_id
        self._rcs_model = rcs_model
        times = [t.timestamp() for t in times]
        positions_xyz = np.stack(
            [
                CoordinateTransformations.geodetic_to_cartesian(*p.as_tuple())
                for p in waypoints
            ],
            axis=0,
        )
        self._f = CubicSpline(times, positions_xyz, extrapolate=False)
        self._v = self._f.derivative()

    def get_monostatic_radars(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Sensor]:
        return []

    def get_pcl_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Sensor]:
        return []

    def get_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        return []

    def get_transmitters(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Transmitter]:
        return []

    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Target]:
        t = (situational_picture.time + dt).timestamp()
        xyz = self._f(t)
        v_xyz = self._v(t)
        lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(*xyz)
        if np.isnan((lat, lon, alt)).any():
            # Out-of-bounds time.
            return []
        else:
            return [
                Target(
                    id=self._target_id,
                    point=Point(lat=lat, lon=lon, alt=alt),
                    cross_section_model=self._rcs_model,
                    velocity=Velocity(vx=v_xyz[0], vy=v_xyz[1], vz=v_xyz[2]),
                )
            ]

    @staticmethod
    def from_trajectory(trajectory: Trajectory) -> WaypointTargetController:
        points: list[Point] = []
        for lat, lon, alt in zip(
            trajectory.lats, trajectory.lons, trajectory.alts, strict=True
        ):
            points.append(Point(lat=lat, lon=lon, alt=alt))
        return WaypointTargetController(
            trajectory.target_id,
            trajectory.times,
            points,
            trajectory.cross_section_model,
        )
