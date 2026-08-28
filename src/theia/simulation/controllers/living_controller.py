import datetime

from pydantic import PrivateAttr
import pydantic

from theia.types import (
    Controller,
    Event,
    EventRelais,
    KillEvent,
    SituationalPicture,
    TheiaException,
)


class AlreadyDeadException(TheiaException):
    pass


class LivingController(Controller):
    """
    Wrapper for another controller to render it "alive", meaning it can be killed.

    This class "deactivates" the child controller when a corresponding ``KillEvent``
    is received.
    """

    child: Controller
    target_id: int

    def model_post_init(self, context):
        super().model_post_init(context)
        self._is_alive = True
        self._relais = EventRelais()
        self.child.register_event_listener(self._relais)

    def on_event(self, event: Event):
        if isinstance(event, KillEvent) and event.target_id == self.target_id:
            if self._is_alive:
                self._is_alive = False
        self.child.on_event(event)

    def register_event_listener(self, listener):
        self._relais.register_event_listener(listener)

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        if self._is_alive:
            self.child.update(situational_picture, dt)
            self.monostatic_sensors = (
                self.child.monostatic_sensors if self._is_alive else []
            )
        else:
            self.monostatic_sensors = []
        self.pcl_sensors = self.child.pcl_sensors if self._is_alive else []
        self.targets = self.child.targets if self._is_alive else []
        self.pet_receivers = self.child.pet_receivers if self._is_alive else []
        self.visual_sensors = self.child.visual_sensors if self._is_alive else []
        self.firing_effectors = self.child.firing_effectors if self._is_alive else []
        self.geojson = self.child.geojson if self._is_alive else {}
