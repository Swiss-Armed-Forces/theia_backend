"""Provide a fastapi server for exposing the latest simulation state to external consumers."""

import datetime
from enum import Enum
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pydantic
import shapely

import theia
from theia.config import FRONTEND_URL
from theia.coordinates import CoordinateTransformations
from theia.coverage import calculate_coverage
from theia.detection.pcl import PclDetector, pcl_track_init_update_masks
from theia.grids import LatLonHeightGrid
from theia.radar_equation import calculate_maximum_monostatic_range
from theia.simulation.logging import SituationalPictureBuffer
from theia.simulation.simulation_director import SimulationDirector
from theia.types import Sensor
from theia.util import mask_to_polygon


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
    id: str
    points: list[TrackPoint]


class ExtrapolatedSituationalPicture(pydantic.BaseModel):
    time: datetime.datetime
    friendly_radars: list[Sensor]
    enemy_tracks: list[ExtrapolatedTrack]


class ExtrapolatedGroundtruth(pydantic.BaseModel):
    target_id: int
    points: list[TrackPoint]


class GeoJSONPolygon(pydantic.BaseModel):
    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[list[float]]]

    @classmethod
    def from_shapely(cls, polygon: shapely.Polygon) -> "GeoJSONPolygon":
        geojson = shapely.geometry.mapping(polygon)
        return cls(
            coordinates=[list(map(list, ring)) for ring in geojson["coordinates"]]
        )


class GeoJSONMultiPolygon(pydantic.BaseModel):
    type: Literal["MultiPolygon"] = "MultiPolygon"
    # One extra nesting level: [polygon][ring][point][coordinate]
    coordinates: list[list[list[list[float]]]]

    @classmethod
    def from_shapely(cls, multi: shapely.MultiPolygon) -> "GeoJSONMultiPolygon":
        geojson = shapely.geometry.mapping(multi)
        return cls(
            coordinates=[
                [list(map(list, ring)) for ring in polygon]
                for polygon in geojson["coordinates"]
            ]
        )


GeoJSONGeometry = GeoJSONPolygon | GeoJSONMultiPolygon


class GeoJSONFeature(pydantic.BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry = pydantic.Field(discriminator="type")
    properties: dict[str, Any] = {}

    @classmethod
    def from_shapely(
        cls,
        shape: shapely.Polygon | shapely.MultiPolygon,
        properties: dict[str, Any] = {},
    ) -> "GeoJSONFeature":
        if isinstance(shape, shapely.Polygon):
            geometry = GeoJSONPolygon.from_shapely(shape)
        elif isinstance(shape, shapely.MultiPolygon):
            geometry = GeoJSONMultiPolygon.from_shapely(shape)
        else:
            raise TypeError(f"Unsupported geometry type: {type(shape)}")
        return cls(geometry=geometry, properties=properties)


def create_app(
    buffer: SituationalPictureBuffer,
    director: SimulationDirector,
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

    @app.post("/calculate_monostatic_coverage")
    def calculate_monostatic_coverage(
        radar: Sensor,
        target_alt: float,
        rcs: float,
        probability_threshold: float,
        azimuth_resolution_degree: float,
    ) -> GeoJSONFeature:
        max_dist = calculate_maximum_monostatic_range(
            radar=radar,
            target_rcs=rcs,
            probability_threshold=probability_threshold,
        )
        polygon = calculate_coverage(
            radar.receiver.point,
            max_dist,
            target_alt,
            d_theta=azimuth_resolution_degree,
        )
        # print(polygon)
        # return
        return GeoJSONFeature(
            geometry=GeoJSONPolygon.from_shapely(polygon),
            properties={"name": "my polygon"},
        )

    @app.post("/calculate_pcl_coverage")
    def calculate_pcl_coverage(
        sensors: list[Sensor],
        grid: LatLonHeightGrid,
        rcs: float,
        snr_threshold: float = theia.config.SNR_THRESHOLD_PCL,
        doppler_threshold: float = theia.config.DOPPLER_SHIFT_THRESHOLD_PCL,
        delay_threshold: float = theia.config.DELAY_THRESHOLD_PCL,
    ) -> tuple[GeoJSONFeature, GeoJSONFeature]:
        """
        Calculate PCL coverage.

        Parameters
        ----------
        sensor: Sensor
            Sensor
        grid: LatLonHeightGrid
            Calculation grid
        rcs: float
            Radar cross section for which to calculate the coverage
        snr_threshold: float, default theia.config.SNR_THRESHOLD_PCL
            Minimum detectable threshold [dB]
        doppler_threshold: float, default theia.config.DOPPLER_SHIFT_THRESHOLD_PCL
            Minimum detectable Doppler shift [Hz]
        delay_threshold: float, default theia.config.DELAY_THRESHOLD_PCL
            Delay threshold for PCL [us].
            This is used to judge whether a given transmitter - target - receiver geometry
            is in the bistatic or the forward scattering regime.

        Returns
        -------
        track_init_coverage: GeoJSONFeature
            Region in which a track init can happen only using PCL
        track_update_coverage: GeoJSONFeature
            Region in which a track update can happen only using PCL
        """
        assert grid.altitude_values.shape[0] == 1

        detector = PclDetector(
            snr_threshold=snr_threshold,
            doppler_threshold=doppler_threshold,
            delay_threshold=delay_threshold,
        )

        track_init_mask, track_update_mask = pcl_track_init_update_masks(
            detector,
            sensors,
            grid,
            rcs,
        )

        polygons_init = mask_to_polygon(
            track_init_mask[:, :, 0],
            grid.latitude_values[0],
            grid.latitude_values[1] - grid.latitude_values[0],
            grid.longitude_values[0],
            grid.longitude_values[1] - grid.longitude_values[0],
        )

        polygons_update = mask_to_polygon(
            track_update_mask[:, :, 0],
            grid.latitude_values[0],
            grid.latitude_values[1] - grid.latitude_values[0],
            grid.longitude_values[0],
            grid.longitude_values[1] - grid.longitude_values[0],
        )
        return (
            GeoJSONFeature(
                geometry=GeoJSONMultiPolygon.from_shapely(
                    shapely.MultiPolygon(polygons_init)
                )
            ),
            GeoJSONFeature(
                geometry=GeoJSONMultiPolygon.from_shapely(
                    shapely.MultiPolygon(polygons_update)
                )
            ),
        )

    @app.get("/health")
    def check_health():
        return "OK"

    @app.post("/pause")
    def pause():
        director.pause()

    @app.post("/resume")
    def resume():
        director.resume()

    @app.get("/is_paused")
    def is_paused() -> bool:
        return director.is_paused()

    @app.get("/speedup")
    def get_speedup_factor() -> float:
        return buffer.get_simulator().get_speedup()

    @app.post("/speedup")
    def set_speedup_factor(speedup_factor: float):
        buffer.get_simulator().set_speedup(speedup_factor)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_URL],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
