from dataclasses import dataclass
import datetime

from theia.types import Controller, SituationalPicture, VisualSensor


@dataclass
class StationaryEyeController(Controller):
    """
    Stationary visual sensor that cannot be detected.

    Useful to implement prior knowledge such as location of infrastructure or
    intel of various sources.
    """

    sensor: VisualSensor

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        if len(self.visual_sensors) == 0:
            self.visual_sensors = [self.sensor]

    def __post_init__(self):
        super().__init__()

    def on_event(self, event):
        pass
