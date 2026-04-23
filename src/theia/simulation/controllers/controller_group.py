import datetime
import itertools
from theia.types import (
    Controller,
    MonostaticSensor,
    PclSensor,
    Receiver,
    SituationalPicture,
)


class ControllerGroup(Controller):
    def __init__(self, controllers: list[Controller]):
        self._controllers = controllers

    def get_monostatic_radars(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[MonostaticSensor]:
        all_radars = [
            c.get_monostatic_radars(situational_picture, dt) for c in self._controllers
        ]
        return list(itertools.chain.from_iterable(all_radars))

    def get_pcl_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[PclSensor]:
        all_radars = [
            c.get_pcl_sensors(situational_picture, dt) for c in self._controllers
        ]
        return list(itertools.chain.from_iterable(all_radars))

    def get_targets(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Receiver]:
        all_targets = [
            c.get_targets(situational_picture, dt) for c in self._controllers
        ]
        return list(itertools.chain.from_iterable(all_targets))
