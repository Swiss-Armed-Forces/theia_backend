import datetime
import functools
import itertools

from theia.types import (
    AbstractEventListener,
    Controller,
    Event,
    EventRelais,
    SituationalPicture,
)


class ControllerGroup(Controller, AbstractEventListener):
    def __init__(self, controllers: list[Controller]):
        super().__init__()
        self._controllers = controllers
        self._relais = EventRelais()
        for c in controllers:
            c.register_event_listener(self._relais)

    def update(self, situational_picture: SituationalPicture, dt: datetime.datetime):
        for child in self._controllers:
            child.update(situational_picture, dt)
        self.monostatic_sensors = itertools.chain.from_iterable(
            [child.monostatic_sensors for child in self._controllers]
        )
        self.pcl_sensors = itertools.chain.from_iterable(
            [child.pcl_sensors for child in self._controllers]
        )
        self.targets = itertools.chain.from_iterable(
            [child.targets for child in self._controllers]
        )
        self.pet_receivers = itertools.chain.from_iterable(
            [child.pet_receivers for child in self._controllers]
        )
        self.visual_sensors = itertools.chain.from_iterable(
            [child.visual_sensors for child in self._controllers]
        )
        self.firing_effectors = itertools.chain.from_iterable(
            [child.firing_effectors for child in self._controllers]
        )
        self.geojson = functools.reduce(
            lambda acc, d: acc | d,
            [child.geojson for child in self._controllers],
            {},
        )

    def register_event_listener(self, listener):
        self._relais.register_event_listener(listener)

    def on_event(self, event: Event):
        for controller in self._controllers:
            controller.on_event(event)
