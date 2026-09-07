import datetime

import pydantic

from theia.types import (
    Controller,
    Event,
    EventRelais,
    GeoJSONFeature,
    SituationalPicture,
)


class GeoJsonController(Controller, pydantic.BaseModel):
    """wrapper that adds GeoJSON to any controller"""

    child: Controller
    geojson_features: dict[str, GeoJSONFeature]

    def model_post_init(self, context):
        super().model_post_init(context)
        self._relais = EventRelais()
        self.child.register_event_listener(self._relais)

    def on_event(self, event: Event):
        self.child.on_event(event)

    def register_event_listener(self, listener):
        self._relais.register_event_listener(listener)

    def add_controller(self, controller: Controller):
        self.child.add_controller(controller)

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        self.child.update(situational_picture, dt)
        self.monostatic_sensors = self.child.monostatic_sensors
        self.pcl_sensors = self.child.pcl_sensors
        self.targets = self.child.targets
        self.pet_receivers = self.child.pet_receivers
        self.visual_sensors = self.child.visual_sensors
        self.firing_effectors = self.child.firing_effectors
        self.geojson = self.child.geojson | self.geojson_features
