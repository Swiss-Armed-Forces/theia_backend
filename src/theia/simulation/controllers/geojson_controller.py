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

    def get_monostatic_radars(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        return self.child.get_monostatic_radars(situational_picture, dt)

    def get_pcl_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        return self.child.get_pcl_sensors(situational_picture, dt)

    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        return self.child.get_targets(situational_picture, dt)

    def get_pet_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        return self.child.get_pet_receivers(situational_picture, dt)

    def get_firing_effectors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        return self.child.get_firing_effectors(situational_picture, dt)

    def get_geojson(self) -> dict[str, GeoJSONFeature]:
        return self.child.get_geojson() | self.geojson_features
