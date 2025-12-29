import abc
import datetime
import enum
from typing import Iterable, Self
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

    @property
    def lat(self) -> float:
        return self.point.lat

    @property
    def lon(self) -> float:
        return self.point.lon

    @property
    def alt(self) -> float:
        return self.point.alt


class Target(pydantic.BaseModel):
    id: int
    point: Point
    cross_section: float  # [m^2] ?
    vlon: float  # [m / s]
    vlat: float  # [m / s]
    vz: float  # [m / s]

    @property
    def lat(self) -> float:
        return self.point.lat

    @property
    def lon(self) -> float:
        return self.point.lon

    @property
    def alt(self) -> float:
        return self.point.alt


class Trajectory(pydantic.BaseModel):
    target_id: int
    times: list[datetime.datetime]
    """Ordered list of times at which the trajectory is sampled."""
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
