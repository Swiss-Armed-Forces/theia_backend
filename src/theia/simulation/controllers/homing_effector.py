import datetime
import math

from scipy.optimize import minimize

from theia.config import SIDC, UNKNOWN_ID
from theia.coordinates import CoordinateTransformations
from theia.distance import line_of_sight_distance
from theia.effectors import DirectFireEffector
from theia.terrain import AbstractTerrainModel
from theia.types import (
    AbstractEffector,
    ConcreteRcsModel,
    Controller,
    KillEvent,
    Point,
    Shot,
    SituationalPicture,
    Target,
    Velocity,
)


class HomingSystem(Controller):
    """
    A target-seeking effector.

    The behaviour is to adjust the trajectory in order to approach an assigned
    track with a constant velocity. The system will try to get as close as
    possible to its target until it runs out of fuel.
    The system commits suicide if it runs out of fuel or the assigned track disappears.

    Loitering behaviour is not implemented in this class, but can be realised
    via a target assignment wrapper class (not implemented as of ``v0.2.0``).

    Examples of such systems are PAC-2/3 missiles or interceptor drones such as
    Sting.

    Assumptions:
    - Constant velocity; the same in all directions,
      i. e. climbing / descending is the same as turning left / right.
    - Constant fuel consumption.
    - The homing effector will try to get as close to the assigned target as
      possible, even if it cannot be reached.
    """

    target_id: int
    """Target ID of the system itself"""
    sidc: SIDC
    speed: float
    """Cruise speed [m / s]"""
    max_dist: float
    """Maximum distance the effector can travel before committing suicide [m]"""
    point: Point
    """Position"""
    rcs: ConcreteRcsModel
    effector: DirectFireEffector
    assigned_track_id: int
    """The tracked target to be destroyed"""
    terrain: AbstractTerrainModel
    # Do not implement the radar sensor yet (KISS and YAGNI principles).
    # radar_sensor: MonostaticSensor | None
    name: str = ""
    travelled_dist: float = 0
    """Distance travelled so far [m]"""

    def on_event(self, event):
        if isinstance(event, Shot) and event.shooter.id == self.effector.id:
            # Commit suicide.
            self._commit_suicide(event.time)

    def update(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        p = self._get_next_position(situational_picture, dt)
        if p is None:
            return
        self.point = p
        self.targets = self._get_targets()
        self.firing_effectors = self._get_firing_effectors(situational_picture, dt)

    def _commit_suicide(self, time: datetime.datetime):
        self._broadcast_event(
            KillEvent(
                id=UNKNOWN_ID,
                time=time,
                target_id=self.target_id,
            )
        )

    def _get_next_position(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> Point | None:
        t = situational_picture.time
        seconds_left = (self.max_dist - self.travelled_dist) / self.speed
        if seconds_left < dt.seconds:
            self._commit_suicide(t)
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
            self._commit_suicide(t)
            return None

        # Position at the time of the situational picture t.
        pos_x, pos_y, pos_z = CoordinateTransformations.geodetic_to_cartesian(
            self.point.lat,
            self.point.lon,
            self.point.alt,
        )

        def pseudo_distance_of_approach(seconds_in_future: float) -> float:
            """
            Assume the current time is t0 = 0 and the future time is t.
            The target's future position P(t) is obtained via
            extrapolation. It will have distance vector
                P(t) = p + R(t) d(t), ||d(t)|| = 1
            from the homing system's current position p.
            We would like to find the distance of closest approach at time t, which
            is given by
                p(t) = p + r(t) * d(t), r(t) := min(t * speed, R(t)).
            The minimum term in the radius is needed to avoid overshooting.

            The following cases are possible:
            1. r^2(t) < R^2(t)
                We do not reach the target, even assuming perfect planning.
            2. r^2(t) == R^2(t)
                We reach the target exactly. The distance is zero.

            Returns
            -------
            float
                R^2 - r^2, i. e. a number >= 0.
            """
            x, vx, y, vy, z, vz = track(
                t + datetime.timedelta(seconds=seconds_in_future[0])
            )
            R2 = (pos_x - x) ** 2 + (pos_y - y) ** 2 + (pos_z - z) ** 2
            r2 = min(( seconds_in_future[0] * self.speed) ** 2, R2)
            return R2 - r2

        result = minimize(
            pseudo_distance_of_approach,
            x0=[dt.seconds],
            bounds=[(dt.seconds, seconds_left)],
            tol=1e-6,
        )
        t_closest = result.x
        DEBUG = False
        if DEBUG:
            print("t_closest = ", t_closest)
            x, vx, y, vy, z, vz = track(
                t + datetime.timedelta(seconds=t_closest[0])
            )
            print("Closest point = ", x, y, z)
        x, vx, y, vy, z, vz = track(t + datetime.timedelta(seconds=t_closest[0]))
        diff = (x - pos_x, y - pos_y, z - pos_z)
        # Avoid division by zero.
        R = max(math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2), 1e-6)
        direction = (diff[0] / R, diff[1] / R, diff[2] / R)
        # Important: We make a tiny step towards the future expected closest approach.
        r = min(dt.seconds * self.speed, R)
        print("Dist to closest = ", R)
        print("Final dist = ", r)

        lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(
            pos_x + r * direction[0],
            pos_y + r * direction[1],
            pos_z + r * direction[2],
        )
        p_next = Point(lat=lat, lon=lon, alt=alt)
        print(p_next)
        return p_next

    def _get_targets(self) -> list[Target]:
        return [
            Target(
                id=self.target_id,
                is_stationary=False,
                name=self.name,
                sidc=self.sidc,
                point=self.point,
                cross_section_model=self.rcs,
                velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
                receiver=None,
                transmitter=None,
            ),
        ]

    def _get_firing_effectors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[tuple[AbstractEffector, Point]]:
        track = next(
            (
                track
                for track in situational_picture.enemy_targets
                if track.id == self.assigned_track_id
            ),
            None,
        )
        if track is None:
            return []

        x, vx, y, vy, z, vz = track(situational_picture.time + dt)
        p_target = CoordinateTransformations.cartesian_to_geodetic(x, y, z)
        p_target = Point(lat=p_target[0], lon=p_target[1], alt=p_target[2])

        d = line_of_sight_distance(
            p_target.lat,
            p_target.lon,
            p_target.alt,
            self.point.lat,
            self.point.lon,
            self.point.alt,
        )
        if d <= self.effector.combat_range and self.terrain.has_line_of_sight(
            p_target, self.point
        ):
            self.effector.point = self.point
            return [(self.effector, p_target)]
        else:
            return []
