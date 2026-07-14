from dataclasses import dataclass

from theia.types import Controller, VisualSensor


@dataclass
class StationaryEyeController(Controller):
    """
    Stationary visual sensor that cannot be detected.

    Useful to implement prior knowledge such as location of infrastructure or
    intel of various sources.
    """

    sensor: VisualSensor

    def __post_init__(self):
        super().__init__()

    def on_event(self, event):
        pass

    def get_monostatic_radars(self, situational_picture, dt):
        return []

    def get_pcl_sensors(self, situational_picture, dt):
        return []

    def get_targets(self, situational_picture, dt):
        return []

    def get_pet_receivers(self, situational_picture, dt):
        return []

    def get_firing_effectors(self, situational_picture, dt):
        return []

    def get_visual_sensors(self, situational_picture, dt) -> list[VisualSensor]:
        return [self.sensor]
