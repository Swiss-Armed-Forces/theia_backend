import abc
import datetime
import enum
from typing import Iterable, Optional, Self
import numpy as np
import pydantic


class RadioClimate(enum.Enum):
    EQUATORIAL = 1
    CONTINENTAL_SUBTROPICAL = 2
    MARITIME_TROPICAL = 3
    DESERT = 4
    CONTINENTAL_TEMPERATE = 5
    MARITIME_TEMPERATE_OVER_LAND = 6
    MARITIME_TEMPERATE_OVER_SEA = 7


class Polarization(enum.Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Point(pydantic.BaseModel):
    lat: float
    """Latitude [decimal °]"""
    lon: float
    """Longitude [decimal °]"""
    alt: float
    """Altitude (meters above sea level) [m]"""


class AttenuationModel(pydantic.BaseModel):
    attenuation_table_angles: list[float]
    attenuation_table_values: list[float]

    @staticmethod
    def _vertical_attenuation_half_wave_dipole(theta):
        """
        Calculate the vertical attenuation of a half-wave dipole antenna.

        Parameters
        -----------
        theta: np.typing.ArrayLike
            Elevation angle [rad] (0rad == horizontal plane)

        Returns
        -------
            Vertical attenuation at the given elevation angle [dB]
        """
        if abs(abs(theta) - np.pi / 2) < 1e-5:
            attenuation_factor_db = 100
        else:
            rad_power_factor = np.square(
                np.cos(np.pi / 2 * np.cos(np.pi / 2 + theta))
                / np.sin(np.pi / 2 + theta)
            )
            attenuation_factor_db = -10.0 * np.log10(rad_power_factor)

        return attenuation_factor_db

    def __call__(self, angles: np.typing.ArrayLike) -> float:
        """
        Interpolate the attenuation table linearly at the given angle.

        Parameters
        ----------
        angles: np.typing.ArrayLike
            Angles in [0, 2 pi] at which to evaluate the attenuation table [rad]

        Returns
        -------
        attenuation: np.typing.ArrayLike
            Attenuation values at the given angles [dB];
            has same shape as ``angles``
        """
        if len(self.attenuation_table_angles) == 0:
            return self._vertical_attenuation_half_wave_dipole(angles)
        else:
            return np.interp(
                angles,
                self.attenuation_table_angles,
                self.attenuation_table_values,
                left=np.nan,
                right=np.nan,
            )


class Radar(pydantic.BaseModel):
    id: int
    """Unique ID"""
    point: Point
    """Coordinates of the transmitter/receiver"""
    power: int
    """Power [W]"""
    erp: float
    """Effective radiated power [W]"""
    antenna_height: float
    """Antenna height [m]"""
    diameter: float
    """Antenna diameter [m]"""
    frequency: float
    """Signal frequency [MHz]"""
    pulse_width: float
    """Pulse width [us]"""
    cpi_pulses: int
    bandwidth: int
    """Band width [MHz]"""
    pfa: float
    """Probability of false alarm (in [0, 1])"""
    min_elevation: float
    """Minimum elevation [°]"""
    max_elevation: float
    """Maximum elevation [°]"""
    rotation_time: float
    """Rotation time [s]"""
    polarization: Polarization
    gain: float = 0
    """Antenna gain [dB]"""
    losses: float = 0
    """antenna to receiver input [dB]"""
    noise_temperature: float = 300.0
    """Receiving system noise temperature [K]"""
    max_coherent_integration_time: float = 0.5
    """
    maximum coherent integration time in [s]
    (use appropriate values for different signals)
    """
    vertical_attenuation: Optional[AttenuationModel] = None
    horizontal_attenuation: Optional[AttenuationModel] = None

    @property
    def lat(self) -> float:
        """Latitude [decimal °]"""
        return self.point.lat

    @property
    def lon(self) -> float:
        """Longitude [decimal °]"""
        return self.point.lon

    @property
    def alt(self) -> float:
        """Altitude (meters above sea level) [m]"""
        return self.point.alt

    @property
    def processing_gain(self) -> float:
        """Processing gain [dB]"""
        return 10 * np.log10(self.max_coherent_integration_time_fm * self.bandwidth)


class Target(pydantic.BaseModel):
    id: int
    """Unique identifier"""
    point: Point
    """Coordinates"""
    cross_section: float
    """Radar cross section [m^2]"""
    vlon: float
    """Velocity in latitude direction [m / s]"""
    vlat: float
    """Velocity in longitude direction [m / s]"""
    vz: float
    """Velocity in radial direction [m / s]"""

    @property
    def lat(self) -> float:
        return self.point.lat

    @property
    def lon(self) -> float:
        return self.point.lon

    @property
    def alt(self) -> float:
        return self.point.alt

    @property
    def speed(self) -> float:
        """Magnitude of velocity vector [m / s]"""
        return np.sqrt(np.square(self.vlat) + np.square(self.vlon) + np.square(self.vz))


class Trajectory(pydantic.BaseModel):
    target_id: int
    times: list[datetime.datetime]
    """Ordered list of times at which the trajectory's waypoints are defined."""
    lats: list[float]
    """Latitude coordinates [°]"""
    lons: list[float]
    """Longitude coordinates [°]"""
    alts: list[float]
    """Altitude coordinates [m above sea level]"""
    vlats: list[float]
    """Velocity along latitude [m / s]"""
    vlons: list[float]
    """Velocity along longitude [m / s]"""
    vzs: list[float]
    """Velocity in altitude [m / s]"""
    cross_sections: list[float]
    """Target cross sections [m^2]"""

    @pydantic.model_validator(mode="after")
    def check_same_length(self) -> Self:
        if (
            len(self.times) != len(self.lats)
            or len(self.times) != len(self.lons)
            or len(self.times) != len(self.alts)
            or len(self.times) != len(self.vlats)
            or len(self.times) != len(self.vlons)
            or len(self.times) != len(self.vzs)
            or len(self.times) != len(self.cross_sections)
        ):
            raise ValueError("Properties are not of same length")
        return self

    @pydantic.model_validator(mode="after")
    def check_times_ordered(self) -> Self:
        if len(self.times) <= 1:
            return self
        for i in range(1, len(self.times)):
            if self.times[i - 1] > self.times[i]:
                raise ValueError("Time steps are not ordered")
        return self


class TargetSimulator(abc.ABC):
    @abc.abstractmethod
    def get_targets(self, time: datetime.datetime) -> Iterable[Target]:
        """Return an iterator over targets at the given time."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_minimum_time(self) -> datetime.datetime:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_maximum_time(self) -> datetime.datetime:
        raise NotImplementedError()


class RadarSimulator(abc.ABC):
    @abc.abstractmethod
    def get_all_radars(self) -> list[Radar]:
        """Return a list of all radars that might be active at some point."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_radars(self, time: datetime.datetime) -> Iterable[Radar]:
        """Return an iterator over radars at the given time."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_minimum_time(self) -> datetime.datetime:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_maximum_time(self) -> datetime.datetime:
        raise NotImplementedError()


class ActiveRadarDetection(pydantic.BaseModel):
    detection_id: int
    time: datetime.datetime
    radar: Radar
    target: Target


class PassiveRadarDetection(pydantic.BaseModel):
    detection_id: int
    time: datetime.datetime
    transmitter: Radar
    receiver: Radar
    target: Target
