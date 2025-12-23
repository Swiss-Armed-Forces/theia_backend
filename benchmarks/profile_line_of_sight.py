from theia.line_of_sight import has_line_of_sight
from theia.detection.active import get_rad_pd
from theia.types import Radar, Target, Polarization, Point


transmitter = Radar(
    id=585,
    point=Point(lat=47.36700085728634, lon=8.537724304199216, alt=407.83600886023686),
    power=20000,
    erp=800,
    antenna_height=10.0,
    diameter=2.0,
    frequency=1000.0,
    pulse_width=1,
    cpi_pulses=1,
    bandwidth=1,
    pfa=1e-6,
    min_elevation=-20.0,
    max_elevation=60.0,
    rotation_time=10.0,
    polarization=Polarization.HORIZONTAL,
)

target = Target(
    id=1,
    point=Point(lat=47.348, lon=8.6266, alt=1000.0),
    cross_section=1.0,
    vlon=250.0,
    vlat=0.0,
    vz=0.0,
)

get_rad_pd(transmitter, target)

for _ in range(10_000):
    has_los = has_line_of_sight(transmitter.point, target.point, 30.0)
