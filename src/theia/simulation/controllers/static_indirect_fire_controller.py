import datetime
import itertools
from typing import Optional


from theia.config import SIDC
from theia.coordinates import CoordinateTransformations
from theia.simulation.controllers.homing_effector import HomingSystem
from theia.simulation.controllers.living_controller import LivingController
from theia.types import (
    ConstantRcsModel,
    Controller,
    Entity,
    IdProvider,
    Point,
    SituationalPicture,
    Target,
    TextEvent,
    Velocity,
)


class StaticIndirectFireController(Controller, arbitrary_types_allowed=True):
    """
    Controller representing a static (i. e. non-moving) indirect fire effector.
    Indirect fire means that instead of attacking the target immediately like
    with small calibre weapons, a projectile is deployed.

    Launch condition: The target's extrapolated track approaches closer than
    the launch distance, which could be the effector range minus a safety margin.

    A real-world example for such an effector is the IRIS-T SL or the
    MIM-104 Patriot launcher families.

    This controller attacks only the assigned track.

    Assumptions:
    - This controller can only launch one projectile per iteration.
    - No projectile is fired as long as there is already a flying projectile for
      the assigned track ID.
    """

    target_id: int
    sidc: SIDC
    rcs: float
    """Radar cross section [m^2]"""
    projectile: HomingSystem
    id_provier: IdProvider
    n_shots_left: int
    launch_distance: float
    """Maximum distance a target is allowed to have to be shot [m]"""
    assigned_track_id: Optional[str] = None
    target_name: str = ""
    geojson_range_altitudes: list[float] = []

    def model_post_init(self, context):
        self._children: list[LivingController[HomingSystem]] = []

    def on_event(self, event):
        for child in self._children:
            child.on_event(event)

    def update(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        # Remove dead projectiles.
        self._children = [child for child in self._children if child._is_alive]
        for child in self._children:
            child.update(situational_picture, dt)
        self.targets = self._get_targets()
        projectile = self._launch_projectile(situational_picture, dt)
        if projectile is not None:
            self_description = f"Launcher #{self.target_id}"
            self._broadcast_event(
                TextEvent(
                    id=-1,
                    time=situational_picture.time + dt,
                    text=f"{self_description} ⤼ Track #{self.assigned_track_id} (projectile #{projectile.target_id})",
                )
            )
            self._children.append(
                LivingController(child=projectile, target_id=projectile.target_id)
            )

    @property
    def point(self) -> Point:
        return self.projectile.point

    def _get_targets(self) -> list[Target]:
        return [
            Target(
                id=self.target_id,
                is_stationary=True,
                name=self.target_name,
                sidc=self.sidc,
                point=self.point,
                cross_section_model=ConstantRcsModel(rcs=self.rcs),
                velocity=Velocity(vx=0, vy=0, vz=0),
                receiver=None,
                transmitter=None,
            )
        ] + list(itertools.chain.from_iterable([child.targets for child in self._children]))

    def _launch_projectile(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> HomingSystem | None:
        if (
            self.assigned_track_id is None
            or self.n_shots_left == 0
            or any(
                child.child.assigned_track_id == self.assigned_track_id
                for child in self._children
            )
        ):
            return None

        track = next(
            (
                track
                for track in situational_picture.enemy_targets
                if track.id == self.assigned_track_id
            ),
            None,
        )
        if track is None:
            return None

        # Determining when to launch a missile is a complex problem well studied
        # in literature (e. g. https://arxiv.org/abs/2311.11905).
        # Instead of performing complex computations, we rely on a simple
        # heuristic: Whenever a track approaches at least to the launch distance,
        # a projectile is launched.
        x, vx, y, vy, z, vz = track(situational_picture.time + dt)
        px, py, pz = CoordinateTransformations.geodetic_to_cartesian(
            self.point.lat,
            self.point.lon,
            self.point.alt,
        )

        d2 = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
        if d2 > self.launch_distance**2:
            return None

        projectile = self.projectile.model_copy(deep=True)
        projectile.assigned_track_id = self.assigned_track_id
        projectile.target_id = self.id_provier.increment(Entity.TARGET)
        self.n_shots_left -= 1
        return projectile
