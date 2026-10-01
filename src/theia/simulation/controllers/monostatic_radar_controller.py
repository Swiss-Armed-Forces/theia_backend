import datetime

from theia.types import (
    ConstantRcsModel,
    Controller,
    Event,
    MonostaticSensor,
    Party,
    SituationalPicture,
    Target,
    TargetInfos,
    Velocity,
)


class MonostaticRadarController(Controller):
    def __init__(
        self,
        target_id: int,
        radar: MonostaticSensor,
        is_blue: bool,
        rcs_model: ConstantRcsModel,
        name: str = "",
    ):
        """
        Parameters
        ----------
        target_id: int
            ID to be used to represent this radar as a detectable target
        radar: MonostaticSensor
            Underlying sensor
        is_blue: bool
            Whether target is BLUE (true); otherwise it is RED
        rcs_model: ConstantRcsModel
            Model for the radar cross section
        name: str
            Human-readable name of the sensor
        """
        super().__init__()
        self._radar = radar
        self._target_id = target_id
        self._name = name
        self._info = TargetInfos.RADAR.with_party(Party.from_is_blue(is_blue))
        self._rcs_model = rcs_model

    def update(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        if len(self.monostatic_sensors) == 0:
            # Only needed at initialisation.
            self.monostatic_sensors = [self._radar]
            self.targets = [
                Target(
                    id=self._target_id,
                    is_stationary=True,
                    name=self._name,
                    info=self._info,
                    point=self._radar.receiver.point,
                    cross_section_model=self._rcs_model,
                    velocity=Velocity(vx=0, vy=0, vz=0),
                    receiver=self._radar.receiver,
                    transmitter=self._radar.transmitter,
                )
            ]

    def on_event(self, event: Event):
        pass
