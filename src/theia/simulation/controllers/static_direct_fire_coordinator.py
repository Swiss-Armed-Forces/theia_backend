import datetime

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.distance import line_of_sight_distance
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.static_direct_fire_controller import (
    StaticDirectFireController,
)
from theia.terrain import AbstractTerrainModel
from theia.types import Point


class StaticDirectFireCoordinator(ControllerGroup):
    """Assigns the closest track to every effector."""

    def __init__(
        self,
        controllers: list[StaticDirectFireController],
        terrain: AbstractTerrainModel,
    ):
        super().__init__(controllers)
        self._terrain = terrain

    def get_firing_effectors(self, situational_picture, dt):
        tracks = situational_picture.enemy_targets
        for c in self._controllers:
            c: StaticDirectFireController = c
            effector = c.effector
            min_track_id: str | None = None
            min_d = np.inf
            for i, track in enumerate(tracks):
                x, vx, y, vy, z, vz = track(situational_picture.time)
                lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
                d = line_of_sight_distance(
                    effector.point.lat,
                    effector.point.lon,
                    effector.point.alt,
                    lat,
                    lon,
                    alt,
                )
                if (
                    d <= effector.combat_range
                    and d < min_d
                    and self._terrain.has_line_of_sight(
                        Point(lat=lat, lon=lon, alt=alt), effector.point
                    )
                ):
                    min_track_id = track.id
                    min_d = d
            c.assigned_track_id = min_track_id

        firing_effectors = super().get_firing_effectors(situational_picture, dt)
        return firing_effectors
