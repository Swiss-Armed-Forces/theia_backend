import datetime

from theia.types import (
    Controller,
    Event,
    IntelSensor,
    SituationalPicture,
)


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
