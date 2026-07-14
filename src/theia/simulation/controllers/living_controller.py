from theia.types import Controller, Event, EventRelais, KillEvent, TheiaException


class AlreadyDeadException(TheiaException):
    pass


class LivingController(Controller):
    """
    Wrapper for another controller to render it "alive", meaning it can be killed.

    This class "deactivates" the child controller when a corresponding ``KillEvent``
    is received.
    """

    def __init__(self, child: Controller, target_id: int):
        super().__init__()
        self._child = child
        self._target_id = target_id
        self._is_alive = True
        self._relais = EventRelais()
        child.register_event_listener(self._relais)

    def on_event(self, event: Event):
        """
        Raises
        ------
        AlreadyDeadException
            When the controller is already dead
        """
        if isinstance(event, KillEvent) and event.target_id == self._target_id:
            if not self._is_alive:
                raise AlreadyDeadException(
                    f"Target with ID {self._target_id} is already dead, but event {event.id} kills it."
                )
            else:
                self._is_alive = False
        self._child.on_event(event)

    def register_event_listener(self, listener):
        self._relais.register_event_listener(listener)

    def get_monostatic_radars(self, situational_picture, dt):
        if self._is_alive:
            return self._child.get_monostatic_radars(situational_picture, dt)
        else:
            return []

    def get_pcl_sensors(self, situational_picture, dt):
        if self._is_alive:
            return self._child.get_pcl_sensors(situational_picture, dt)
        else:
            return []

    def get_targets(self, situational_picture, dt):
        if self._is_alive:
            return self._child.get_targets(situational_picture, dt)
        else:
            return []

    def get_pet_receivers(self, situational_picture, dt):
        if self._is_alive:
            return self._child.get_pet_receivers(situational_picture, dt)
        else:
            return []

    def get_firing_effectors(self, situational_picture, dt):
        if self._is_alive:
            return self._child.get_firing_effectors(situational_picture, dt)
        else:
            return []
