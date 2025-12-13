import pydantic


class Radar(pydantic.BaseModel):
    id: int
    lat: float
    lon: float
    power: int
    diameter: float
    frequency: float
    pulse_width: float
    cpi_pulses: int
    bandwidth: int
    pfa: float
    min_elevation: float
    max_elevation: float


class Point(pydantic.BaseModel):
    lat: float
    lon: float
    alt: float
