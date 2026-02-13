import datetime
from theia.types import (
    Radar,
    Receiver,
    ReceiverController,
    SituationalPicture,
    Transmitter,
    TransmitterController,
)


class MonostaticRadarController(ReceiverController, TransmitterController):
    def __init__(self, radar: Radar):
        self._radar = radar

    def get_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        return self._radar.receiver

    def get_transmitters(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Transmitter]:
        return self._radar.transmitter
