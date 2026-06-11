from __future__ import annotations
import datetime
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from theia.coordinates import CoordinateTransformations
from theia.types import (
    AbstractEffector,
    Controller,
    Event,
    MonostaticSensor,
    PclSensor,
    Point,
    RcsModel,
    Receiver,
    SituationalPicture,
    Target,
    Trajectory,
    Velocity,
)


class WaypointTargetController(Controller):
    def __init__(
        self,
        name: str,
        sidc: str,
        target_id: int,
        times: list[datetime.datetime],
        waypoints: list[Point],
        rcs_model: RcsModel,
        radar: Optional[MonostaticSensor] = None,
    ):
        """
        Parameters
        ----------
        name: str
            Name
        sidc: str
            SIDC code of the target
        target_id: int
            Unique identifier
        times: list[datetime.datetime]
            Times at which the waypoints are reached
        waypoints: list[Point]
            Waypoints of the trajectory
        rcs_model: RcsModel
            Radar cross section (RCS) model
        radar: Optional[MonostaticSensor], default None
            On-board radar (if any); is always active if present
            The position of the radar is updated to the target's current position,
            so the "point" property of the transmitter and receiver are ignored
        """
        assert len(times) == len(waypoints)
        self._name = name
        self._sidc = sidc
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
        self._sensor = radar

    def get_monostatic_radars(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[MonostaticSensor]:
        if self._sensor is None:
            return []

        t = (situational_picture.time + dt).timestamp()
        x, y, z = self._f(t)
        lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
        p = Point(lat=lat, lon=lon, alt=alt)
        self._sensor.transmitter.point = p
        self._sensor.receiver.point = p
        return [self._sensor]

    def get_pcl_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[PclSensor]:
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
            rx = None
            tx = None
            if self._sensor is not None:
                rx = self._sensor.receiver.model_copy(deep=True)
                tx = self._sensor.transmitter.model_copy(deep=True)
                p = Point(lat=lat, lon=lon, alt=alt)
                tx.point = p
            return [
                Target(
                    id=self._target_id,
                    is_stationary=False,
                    name=self._name,
                    sidc=self._sidc,
                    point=Point(lat=lat, lon=lon, alt=alt),
                    cross_section_model=self._rcs_model,
                    velocity=Velocity(vx=v_xyz[0], vy=v_xyz[1], vz=v_xyz[2]),
                    transmitter=tx,
                    receiver=rx,
                )
            ]

    def get_pet_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        return []

    @staticmethod
    def from_trajectory(
        trajectory: Trajectory,
        sidc: str,
        name: str = "",
        sensor: Optional[MonostaticSensor] = None,
    ) -> WaypointTargetController:
        points: list[Point] = []
        for lat, lon, alt in zip(
            trajectory.lats, trajectory.lons, trajectory.alts, strict=True
        ):
            points.append(Point(lat=lat, lon=lon, alt=alt))
        return WaypointTargetController(
            name,
            sidc,
            trajectory.target_id,
            trajectory.times,
            points,
            trajectory.cross_section_model,
            sensor,
        )

    def get_effectors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[AbstractEffector]:
        return []

    def on_event(self, event: Event):
        pass
