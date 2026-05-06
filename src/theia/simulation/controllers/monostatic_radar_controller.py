import datetime
from theia.types import (
    Controller,
    MonostaticSensor,
    PclSensor,
    Receiver,
    SituationalPicture,
    Target,
)


class MonostaticRadarController(Controller):
    def __init__(self, radar: MonostaticSensor):
        self._radar = radar

    def get_monostatic_radars(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[MonostaticSensor]:
        return [self._radar]

    def get_pcl_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[PclSensor]:
        return []

    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Target]:
        return []

    def get_pet_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        return []
