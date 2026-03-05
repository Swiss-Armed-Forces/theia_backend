"""Provide a fastapi server for exposing the latest simulation state to external consumers."""

import datetime
from enum import Enum

from fastapi import FastAPI
import pydantic

from theia.coordinates import CoordinateTransformations
from theia.simulation.logging import SituationalPictureBuffer
from theia.types import Radar


class Team(Enum):
    blue = "BLUE"
    red = "RED"


class TrackPoint(pydantic.BaseModel):
    time: datetime.datetime
    """Time coordinate of the trajectory."""
    lat: float
    """Latitude [decimal °]"""
    lon: float
    """Longitude [decimal °]"""
    alt: float
    """Altitude [m above sea level]"""
    v_east: float
    """Velocity along east direction in local ENU frame [m / s]"""
    v_north: float
    """Velocity along north direction in local ENU frame [m / s]"""
    v_up: float
    """Velocity along up direction in local ENU frame [m / s]"""


class ExtrapolatedTrack(pydantic.BaseModel):
    id: int
    points: list[TrackPoint]


class ExtrapolatedSituationalPicture(pydantic.BaseModel):
    time: datetime.datetime
    friendly_radars: list[Radar]
    enemy_tracks: list[ExtrapolatedTrack]


class ExtrapolatedGroundtruth(pydantic.BaseModel):
    target_id: int
    points: list[TrackPoint]


def create_app(
    buffer: SituationalPictureBuffer,
    max_extrapolation_time: datetime.timedelta = datetime.timedelta(minutes=1),
    extrapolation_resolution: datetime.timedelta = datetime.timedelta(seconds=1),
) -> FastAPI:
    app = FastAPI()

    @app.get("/time")
    def get_time() -> datetime.datetime:
        # We assume that the red and blue situational pictures share the same time.
        # This is consistent with the implementation of the main simulation loop.
        # The differences would be negligible, anyway.
        return buffer.get_situational_picture(True).time

    @app.post("/situational_picture/{which}")
    def get_situational_picture(
        which: Team, times: list[datetime.datetime]
    ) -> ExtrapolatedSituationalPicture:
        picture = buffer.get_situational_picture(which == Team.blue)

        extrapolated_tracks: list[ExtrapolatedTrack] = []
        print("N enemies: ", len(picture.enemy_targets))
        for track in picture.enemy_targets:
            states = [track(t).flatten() for t in times]

            track_points = []
            for t, y in zip(times, states, strict=True):
                x, vx, y, vy, z, vz = y
                lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
                # p = Point(lat=lat, lon=lon, alt=alt)
                # v_east, v_north, v_up = CoordinateTransformations.ecef_to_enu(
                #     p,
                #     (vx, vy, vz),
                # )
                track_points.append(
                    TrackPoint(
                        time=t,
                        lat=lat,
                        lon=lon,
                        alt=alt,
                        # TODO: Implement velocity conversion ECEF -> ENU
                        v_east=0.0,
                        v_north=0.0,
                        v_up=0.0,
                    )
                )
            extrapolated_tracks.append(
                ExtrapolatedTrack(id=track.id, points=track_points)
            )
        return ExtrapolatedSituationalPicture(
            time=picture.time,
            friendly_radars=picture.friendly_radars,
            enemy_tracks=extrapolated_tracks,
        )

    @app.post("/ground_truth/{which}")
    def get_ground_truth(
        which: Team, times: list[datetime.datetime]
    ) -> list[ExtrapolatedGroundtruth]:
        trajectories = buffer.get_ground_truth_trajectories(which == Team.blue)
        results = []
        for trajectory in trajectories:
            points = []
            for time in times:
                target = trajectory(time)
                points.append(
                    TrackPoint(
                        time=time,
                        lat=target.lat,
                        lon=target.lon,
                        alt=target.alt,
                        # TODO: Implement velocity conversion ECEF -> ENU
                        v_east=0.0,
                        v_north=0.0,
                        v_up=0.0,
                    )
                )
            results.append(
                ExtrapolatedGroundtruth(
                    target_id=trajectory.target_id,
                    points=points,
                )
            )
        return results

    return app
