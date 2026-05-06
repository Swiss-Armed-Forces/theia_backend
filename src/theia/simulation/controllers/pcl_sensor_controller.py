import datetime

from theia.types import (
    ConstantRcsModel,
    Controller,
    AbstractSensor,
    MonostaticSensor,
    PclSensor,
    Receiver,
    SituationalPicture,
    Target,
    Velocity,
)


class PclSensorController(Controller):
    def __init__(
        self,
        target_id_rx: int,
        target_id_tx: int,
        sensor: MonostaticSensor,
        is_blue: bool,
        rcs_model: ConstantRcsModel,
        name: str = "",
    ):
        self._sensor = sensor
        self._name = name
        self._rcs_model = rcs_model
        self._sidc_rx = f"10-0-{3 if is_blue else 6}-15-0-0-00-220300-00-00"
        self._sidc_tx = f"10-0-{4}-20-0-0-00-121201-00-00"
        self._target_id_rx = target_id_rx
        self._target_id_tx = target_id_tx

    def get_monostatic_radars(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[MonostaticSensor]:
        return []

    def get_pcl_sensors(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[PclSensor]:
        return [self._sensor]

    def get_targets(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[AbstractSensor]:
        return [
            Target(
                id=self._target_id_rx,
                is_stationary=True,
                name=self._name,
                sidc=self._sidc_rx,
                point=self._sensor.receiver.point,
                cross_section_model=self._rcs_model,
                velocity=Velocity(vx=0, vy=0, vz=0),
                receiver=self._sensor.receiver,
                transmitter=None,
            ),
            Target(
                id=self._target_id_tx,
                is_stationary=True,
                name=f"Tx (ID {self._sensor.transmitter})",
                sidc=self._sidc_tx,
                point=self._sensor.receiver.point,
                cross_section_model=self._rcs_model,
                velocity=Velocity(vx=0, vy=0, vz=0),
                receiver=None,
                transmitter=self._sensor.transmitter,
            ),
        ]

    def get_pet_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        return []
