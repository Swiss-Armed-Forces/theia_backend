import datetime

from theia.types import (
    Controller,
    AbstractSensor,
    MonostaticSensor,
    PclSensor,
    SituationalPicture,
)


class PclSensorController(Controller):
    def __init__(self, sensor: PclSensor):
        self._sensor = sensor

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
        return []
