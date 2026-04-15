import datetime
import itertools
from theia.types import (
    Controller,
    Sensor,
    Receiver,
    SituationalPicture,
    Target,
    Transmitter,
)


class ControllerGroup(Controller):
    def __init__(self, controllers: list[Controller]):
        self._controllers = controllers

    def get_monostatic_radars(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Sensor]:
        all_radars = [
            c.get_monostatic_radars(situational_picture, dt) for c in self._controllers
        ]
        return list(itertools.chain.from_iterable(all_radars))

    def get_pcl_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Sensor]:
        all_radars = [
            c.get_pcl_sensors(situational_picture, dt) for c in self._controllers
        ]
        return list(itertools.chain.from_iterable(all_radars))

    def get_transmitters(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Transmitter]:
        all_transmitters = [
            c.get_transmitters(situational_picture, dt) for c in self._controllers
        ]
        return list(itertools.chain.from_iterable(all_transmitters))

    def get_receivers(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Target]:
        all_receivers = [
            c.get_receivers(situational_picture, dt) for c in self._controllers
        ]
        return list(itertools.chain.from_iterable(all_receivers))

    def get_targets(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Receiver]:
        all_targets = [
            c.get_targets(situational_picture, dt) for c in self._controllers
        ]
        return list(itertools.chain.from_iterable(all_targets))
