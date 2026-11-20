import datetime

from theia.config import UNKNOWN_RCS
from theia.types import (
    ConstantRcsModel,
    Controller,
    Entity,
    Event,
    IdProvider,
    IntelSensor,
    Point,
    SituationalPicture,
    TargetInfo,
    Trajectory,
)


def build_stationary_intel_sensor(
    id_provider: IdProvider,
    point: Point,
    target_info: TargetInfo,
) -> IntelSensor:
    """
    Build an intel sensor that reports a stationary target at `point`.

    The trajectory spans all time; a controller limits how long the sensor is
    exposed. The target and sensor IDs come from `id_provider`. The target's
    RCS is unknown.
    """
    t0 = datetime.datetime.min.replace(tzinfo=datetime.UTC)
    t1 = datetime.datetime.max.replace(tzinfo=datetime.UTC)
    trajectory = Trajectory(
        target_id=id_provider.increment(Entity.TARGET),
        target_info=target_info,
        times=[t0, t1],
        lats=[point.lat, point.lat],
        lons=[point.lon, point.lon],
        alts=[point.alt, point.alt],
        vxs=[0.0, 0.0],
        vys=[0.0, 0.0],
        vzs=[0.0, 0.0],
        cross_section_model=ConstantRcsModel(rcs=UNKNOWN_RCS),
    )
    return IntelSensor(id=id_provider.increment(Entity.SENSOR), trajectory=trajectory)


class IntelController(Controller):
    """
    Represents knowledge about an enemy target.

    The controller is stateless. It exposes a single IntelSensor.
    The controller is agnostic to whether the underlying intel is
    true or false. It simply exposes the information.
    """

    sensor: IntelSensor

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        self.intel_sensors = [self.sensor]

    def on_event(self, event: Event):
        pass
