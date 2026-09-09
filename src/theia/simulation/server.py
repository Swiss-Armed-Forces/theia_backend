"""Provide a fastapi server for exposing the latest simulation state to external consumers."""

from __future__ import annotations

import datetime
import os
import pathlib
from enum import Enum
from functools import cache
from typing import Optional

import numpy as np
import pydantic
import shapely
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import theia
from theia.config import FRONTEND_URL, TERRAIN_HBV_DATA_DIR
from theia.coordinates import CoordinateTransformations
from theia.coverage import (
    calculate_coverage,
    pcl_track_init_update_masks_parallel,
)
from theia.data_loading import load_bakom_ukw_transmitters
from theia.detection.pcl import PclDetector
from theia.distance import haversine, line_of_sight_distance
from theia.grids import LatLonHeightGrid
from theia.radar_equation import calculate_maximum_monostatic_range
from theia.simulation.simulation_director import SimulationDirector
from theia.simulation.theia_logging import SituationalPictureBuffer
from theia.types import (
    Event,
    GeoJSONFeature,
    GeoJSONMultiPolygon,
    GeoJSONPolygon,
    MonostaticSensor,
    PclSensor,
    Receiver,
    Sensor,
    Transmitter,
)
from theia.util import mask_to_polygon


class Team(Enum):
    blue = "BLUE"
    red = "RED"


class EventMessage(pydantic.BaseModel):
    time: datetime.datetime
    msg: str

    def from_event(event: Event) -> EventMessage:
        return EventMessage(time=event.time, msg=str(event))


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
    sidc: str
    receiver: Optional[Receiver] = None
    transmitter: Optional[Transmitter] = None


class ExtrapolatedSituationalPicture(pydantic.BaseModel):
    time: datetime.datetime
    friendly_radars: list[Sensor]
    enemy_tracks: list[ExtrapolatedTrack]


class ExtrapolatedGroundtruth(pydantic.BaseModel):
    target_id: int
    name: str
    points: list[TrackPoint]
    sidc: str


class DefaultMonostaticSensorConfiguration(pydantic.BaseModel):
    name: str
    description: str
    sensor: MonostaticSensor


def create_app(
    buffer: SituationalPictureBuffer,
    director: SimulationDirector,
    max_extrapolation_time: datetime.timedelta = datetime.timedelta(minutes=1),
    extrapolation_resolution: datetime.timedelta = datetime.timedelta(seconds=1),
) -> FastAPI:
    app = FastAPI()

    @app.get("/time")
    def get_time() -> datetime.datetime:
        return buffer.get_simulator()._t

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
                ExtrapolatedTrack(
                    id=track.id,
                    points=track_points,
                    sidc=track.sidc,
                )
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
        # TODO: Fix this cleanly. We needed a quick fix due to a deadline, so
        # we left the API as is and mostly simply ignored the times parameter.
        targets = buffer.get_ground_truth_state(which == Team.blue, times[0])
        results: list[ExtrapolatedGroundtruth] = []
        for target in targets:
            results.append(
                ExtrapolatedGroundtruth(
                    target_id=target.id,
                    name=target.name,
                    points=[
                        TrackPoint(
                            time=times[0],
                            lat=target.lat,
                            lon=target.lon,
                            alt=target.alt,
                            v_east=0,
                            v_north=0,
                            v_up=0,
                        )
                    ],
                    sidc=target.sidc,
                )
            )
        return results

    @app.post("/calculate_monostatic_coverage")
    def calculate_monostatic_coverage(
        radar: MonostaticSensor,
        target_alt: float,
        target_rcs: float,
        probability_threshold: float,
        lat_res: float,
        lon_res: float,
    ) -> list[GeoJSONFeature]:
        max_dist = calculate_maximum_monostatic_range(
            radar=radar,
            target_rcs=target_rcs,
            probability_threshold=probability_threshold,
        )
        polygons = calculate_coverage(
            director._simulator._terrain_model,
            radar.receiver.point,
            max_dist,
            target_alt,
            lat_res,
            lon_res,
        )
        return [
            GeoJSONFeature(
                geometry=GeoJSONPolygon.from_shapely(p),
                properties={"name": f"coverage (dlat={lat_res}, dlon={lon_res})"},
            )
            for p in polygons
        ]

    @app.post("/calculate_min_detectable_rcs")
    def calculate_min_detectable_rcs(
        sensor: PclSensor,
        grid: LatLonHeightGrid,
        snr_threshold: float = theia.config.SNR_THRESHOLD_PCL,
        doppler_threshold: float = theia.config.DOPPLER_SHIFT_THRESHOLD_PCL,
        delay_threshold: float = theia.config.DELAY_THRESHOLD_PCL,
    ) -> list[list[list[float]]]:
        """
        Calculate the minimum detectable radar cross section for the given
        sensor on a grid. The grid dimensions are (lat, lon, MASL).

        The value -1 indicates a NaN, i. e. the sensor cannot detect a target
        at all at that position.
        """
        detector = PclDetector(
            snr_threshold=snr_threshold,
            doppler_threshold=doppler_threshold,
            delay_threshold=delay_threshold,
        )
        values = detector.minimum_detectable_rcs_grid(
            sensor.receiver,
            sensor.transmitter,
            grid,
        )
        return np.nan_to_num(values, copy=False, nan=-1)

    @app.post("/calculate_pcl_coverage")
    def calculate_pcl_coverage(
        sensors: list[PclSensor],
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
        sensors: list[PclSensor]
            Sensors
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
        if len(sensors) == 0:
            return GeoJSONFeature.from_shapely(
                shapely.Polygon()
            ), GeoJSONFeature.from_shapely(shapely.Polygon())
        assert grid.altitude_values.shape[0] == 1

        detector = PclDetector(
            snr_threshold=snr_threshold,
            doppler_threshold=doppler_threshold,
            delay_threshold=delay_threshold,
        )

        track_init_mask, track_update_mask = pcl_track_init_update_masks_parallel(
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

    @app.get("/events")
    def get_events(t_start: Optional[datetime.datetime] = None) -> list[EventMessage]:
        no_constraint = t_start is None
        return [
            EventMessage.from_event(e)
            for e in buffer.get_events()
            if no_constraint or e.time >= t_start
        ]

    @app.get("/geojson/{which}")
    def get_geojson(which: Team) -> dict[str, GeoJSONFeature]:
        return buffer.get_geojson(which == Team.blue)

    @app.get("/elevation_at/{lat}_{lon}")
    def elevationAt(lat: float, lon: float) -> float:
        """
        Calculate elevation [MASL] for the given decimal (lat, lon) coordinates.
        """
        return director._simulator._terrain_model.elevationAt(lat, lon)

    @cache
    @app.get("/fm_transmitters")
    def get_fm_transmitters() -> list[Transmitter]:
        return load_bakom_ukw_transmitters()

    @app.get("/line_of_sight_distance/{lat1}_{lon1}_{alt1}/{lat2}_{lon2}_{alt2}")
    def get_los_distance(
        lat1: float,
        lon1: float,
        alt1: float,
        lat2: float,
        lon2: float,
        alt2: float,
    ) -> float:
        """Calculate LOS distance [m]"""
        return line_of_sight_distance(lat1, lon1, alt1, lat2, lon2, alt2)

    @app.get("/haversine_distance/{lat1}_{lon1}/{lat2}_{lon2}")
    def get_haversine_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        return haversine(lon1, lat1, lon2, lat2)

    @app.get("/terrain_models")
    def get_terrain_models() -> list[str]:
        files = os.listdir(TERRAIN_HBV_DATA_DIR)
        model_names = [file.replace(".zip", "") for file in files]
        return ["SRTM"] + model_names

    @app.get("/default_monostatic_sensor_configurations")
    def get_default_monostatic_sensor_configurations() -> list[
        DefaultMonostaticSensorConfiguration
    ]:
        config_dir = (
            pathlib.Path(__file__).parent.parent.parent.parent
            / "data"
            / "default_configurations"
        )
        results: list[DefaultMonostaticSensorConfiguration] = []
        for filename in os.listdir(config_dir.absolute()):
            with open((config_dir / filename).absolute(), "r") as file:
                results.append(
                    DefaultMonostaticSensorConfiguration.model_validate_json(
                        file.read()
                    )
                )
        return results

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_URL],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
