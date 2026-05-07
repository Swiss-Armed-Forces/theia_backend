import datetime
from theia.types import (
    ConstantRcsModel,
    Controller,
    MonostaticSensor,
    PclSensor,
    Receiver,
    SituationalPicture,
    Target,
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
        self._radar = radar
        self._target_id = target_id
        self._name = name
        self._sidc = f"100{3 if is_blue else 6}1500002203000000"
        self._rcs_model = rcs_model

    def get_monostatic_radars(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[MonostaticSensor]:
        return [self._radar]

    def get_pcl_sensors(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[PclSensor]:
        return []

    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Target]:

        return [
            Target(
                id=self._target_id,
                is_stationary=True,
                name=self._name,
                sidc=self._sidc,
                point=self._radar.receiver.point,
                cross_section_model=self._rcs_model,
                velocity=Velocity(vx=0, vy=0, vz=0),
                receiver=self._radar.receiver,
                transmitter=self._radar.transmitter,
            )
        ]

    def get_pet_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        return []
