import datetime

from theia.types import (
    ConstantRcsModel,
    Controller,
    Event,
    Party,
    PclSensor,
    SituationalPicture,
    Target,
    TargetInfos,
    Velocity,
)


class PclSensorController(Controller):
    def __init__(
        self,
        target_id_rx: int,
        target_id_tx: int,
        sensor: PclSensor,
        is_blue: bool,
        rcs_model: ConstantRcsModel,
        own_receiver: bool,
        own_transmitter: bool,
        name: str = "",
    ):
        super().__init__()
        self._sensor = sensor
        self._name = name
        self._rcs_model = rcs_model
        self._info_rx = TargetInfos.PCL_RECEIVER.with_party(Party.from_is_blue(is_blue))
        self._info_tx = TargetInfos.TRANSMITTER
        self._target_id_rx = target_id_rx
        self._target_id_tx = target_id_tx
        self._own_receiver = own_receiver
        self._own_transmitter = own_transmitter

    def update(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ):
        if len(self.pcl_sensors) == 0:
            # Only needed at initialisation.
            self.pcl_sensors = [self._sensor]
            targets = []
            if self._own_receiver:
                targets.append(
                    Target(
                        id=self._target_id_rx,
                        is_stationary=True,
                        name=self._name,
                        info=self._info_rx,
                        point=self._sensor.receiver.point,
                        cross_section_model=self._rcs_model,
                        velocity=Velocity(vx=0, vy=0, vz=0),
                        receiver=self._sensor.receiver,
                        transmitter=None,
                    )
                )
            if self._own_transmitter:
                targets.append(
                    Target(
                        id=self._target_id_tx,
                        is_stationary=True,
                        name=f"Tx (ID {self._sensor.transmitter})",
                        info=self._info_tx,
                        point=self._sensor.receiver.point,
                        cross_section_model=self._rcs_model,
                        velocity=Velocity(vx=0, vy=0, vz=0),
                        receiver=None,
                        transmitter=self._sensor.transmitter,
                    )
                )
            self.targets = targets

    def on_event(self, event: Event):
        pass
