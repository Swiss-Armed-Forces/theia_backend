import datetime

import pydantic

from theia.simulation.controllers.intel_controller import (
    build_stationary_intel_sensor,
)
from theia.types import (
    Controller,
    Event,
    IntelSensor,
    Point,
    SimulationContext,
    SituationalPicture,
    TargetInfo,
)


class KnownTargetController(Controller):
    """
    Lasting knowledge about a stationary target, e.g. an enemy facility.

    Exposes one `IntelSensor` that reports a target at `point` with
    `target_info` for the whole simulation. Like `IntelController`, it is
    agnostic to whether the intel is true: `point` and `target_info` are the
    owning side's belief and need not match any real target.

    The IDs of the reported target and of the sensor come from the
    simulation's `IdProvider` when the controller is bound, so it must be
    bound before its first update.
    """

    point: Point
    target_info: TargetInfo

    _sensor: IntelSensor | None = pydantic.PrivateAttr(default=None)

    def bind(self, context: SimulationContext):
        super().bind(context)
        # Bind may be repeated; the target keeps its IDs.
        if self._sensor is None:
            self._sensor = build_stationary_intel_sensor(
                context.id_provider, self.point, self.target_info
            )

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        # Raises if the controller has not been bound.
        self.simulation_context  # noqa: B018
        self.intel_sensors = [self._sensor]

    def on_event(self, event: Event):
        pass
