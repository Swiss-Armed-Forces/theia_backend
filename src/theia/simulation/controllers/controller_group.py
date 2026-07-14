import datetime
import itertools
from theia.types import (
    AbstractEffector,
    Controller,
    Event,
    MonostaticSensor,
    PclSensor,
    Point,
    Receiver,
    SituationalPicture,
    Target,
    VisualSensor,
)


class ControllerGroup(Controller):
    def __init__(self, controllers: list[Controller]):
        super().__init__()
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
    ) -> list[Target]:
        all_targets = [
            c.get_targets(situational_picture, dt) for c in self._controllers
        ]
        all_targets = list(itertools.chain.from_iterable(all_targets))
        return all_targets

    def get_pet_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        all_receivers = [
            c.get_pet_receivers(situational_picture, dt) for c in self._controllers
        ]
        if len(all_receivers) == 0:
            return []
        else:
            return list(itertools.chain.from_iterable(all_receivers))

    def get_firing_effectors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[tuple[AbstractEffector, Point]]:
        all_shots = [
            c.get_firing_effectors(situational_picture, dt) for c in self._controllers
        ]
        if len(all_shots) == 0:
            return []
        else:
            return list(itertools.chain.from_iterable(all_shots))

    def get_visual_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[VisualSensor]:
        all_sensors = [
            c.get_visual_sensors(situational_picture, dt) for c in self._controllers
        ]
        if len(all_sensors) == 0:
            return []
        else:
            return list(itertools.chain.from_iterable(all_sensors))

    def on_event(self, event: Event):
        for controller in self._controllers:
            controller.on_event(event)
