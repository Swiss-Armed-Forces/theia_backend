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
    lat: float  # [°]
    lon: float  # [°]
    alt: float  # [m]


class Radar(pydantic.BaseModel):
    id: int
    point: Point
    power: int  # [W]
    erp: float  # [W] effective radiated power
    antenna_height: float  # [m]
    diameter: float  # antenna diameter [m]
    frequency: float  # [MHz]
    pulse_width: float  # [us]
    cpi_pulses: int
    bandwidth: int  # [MHz]
    pfa: float
    min_elevation: float  # [°]
    max_elevation: float  # [°]
    rotation_time: float  # [s]
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
