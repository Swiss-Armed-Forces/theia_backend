import datetime

from theia.types import Controller, Sensor, SituationalPicture


class PclSensorController(Controller):
    def __init__(self, sensor: Sensor):
        self._sensor = sensor

    def get_monostatic_radars(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Sensor]:
        return []

    def get_pcl_sensors(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Sensor]:
        return [self._sensor]

    def get_receivers(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Sensor]:
        return [self._sensor.receiver]

    def get_transmitters(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Sensor]:
        return [self._sensor.transmitter]

    def get_targets(
        self, situational_picture: SituationalPicture, dt: datetime.timedelta
    ) -> list[Sensor]:
        return []
