import datetime

from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.types import (
    Controller,
    Event,
    EventRelais,
    SituationalPicture,
)


class SwitchableMonostaticRadarController(Controller):
    """
    Wrapper that lets a radar be turned on and off.

    The wrapped controller is not altered. While the radar is off, this wrapper
    hides its sensor, so it detects nothing, and removes the transmitter from
    its target, so passive emitter tracking (PET) no longer sees it.
    The radar stays a target that can be detected by other means and hit.
    """

    child: MonostaticRadarController
    sensor_id: int
    """ID of the wrapped monostatic sensor"""
    is_on: bool = True

    def model_post_init(self, context):
        super().model_post_init(context)
        self._relais = EventRelais()
        self.child.register_event_listener(self._relais)

    def on_event(self, event: Event):
        self.child.on_event(event)

    def register_event_listener(self, listener):
        self._relais.register_event_listener(listener)

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        self.child.update(situational_picture, dt)
        if self.is_on:
            self.monostatic_sensors = self.child.monostatic_sensors
            self.targets = self.child.targets
        else:
            self.monostatic_sensors = []
            self.targets = [
                t.model_copy(update={"transmitter": None}) for t in self.child.targets
            ]
        self.pcl_sensors = self.child.pcl_sensors
        self.pet_receivers = self.child.pet_receivers
        self.visual_sensors = self.child.visual_sensors
        self.intel_sensors = self.child.intel_sensors
        self.firing_effectors = self.child.firing_effectors
        self.geojson = self.child.geojson
