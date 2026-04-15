import datetime
from theia.types import (
    Controller,
    Radar,
    Receiver,
    SituationalPicture,
    Target,
    Transmitter,
)


class MonostaticRadarController(Controller):
    def __init__(self, radar: Radar):
        self._radar = radar
    
    def get_monostatic_radars(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Radar]:
        return [self._radar]
    
    def get_pcl_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Radar]:
        return []

    def get_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        return [self._radar.receiver]

    def get_transmitters(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Transmitter]:
        return [self._radar.transmitter]

    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Target]:
        return []
