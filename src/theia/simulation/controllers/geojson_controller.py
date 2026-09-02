from dataclasses import dataclass
import datetime

from theia.types import Controller, Event, GeoJSONFeature, SituationalPicture


@dataclass
class GeoJsonController(Controller):
    """wrapper that adds GeoJSON to any controller"""

    child: Controller
    geojson_features: dict[str, GeoJSONFeature]

    def __post_init__(self):
        super().__init__()

    def on_event(self, event: Event):
        self.child.on_event(event)

    def add_controller(self, controller: Controller):
        self.child.add_controller(controller)

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        self.child.update(situational_picture, dt)
        self.geojson = self.child.geojson | self.geojson_features
