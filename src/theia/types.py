import enum
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
