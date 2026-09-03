import datetime

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.distance import line_of_sight_distance
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.static_gbad_controller import StaticGbadController
from theia.terrain import AbstractTerrainModel
from theia.types import Point, SituationalPicture


class StaticGbadCoordinator(ControllerGroup):
    """
    Assigns the closest track to every effector.

    Target assignment is assumed to be instantaneous, i. e. the effector can
    fire in the same turn as it is assigned a new target.
    """

    def __init__(
        self,
        controllers: list[StaticGbadController],
        terrain: AbstractTerrainModel,
    ):
        super().__init__(controllers)
        self._terrain = terrain

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        tracks = situational_picture.enemy_targets
        for c in self._controllers:
            c: StaticGbadController = c
            effector = c.effector
            min_track_id: str | None = None
            min_d = np.inf
            for track in tracks:
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
        super().update(situational_picture, dt)
