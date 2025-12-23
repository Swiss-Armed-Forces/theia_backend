from typing import Generator
import geopy
from geopy.distance import distance
import numba
import numpy as np
from numpy import radians
import pyproj

from theia.coordinates import CoordinateTransformations
from theia.types import Point


R_EARTH = 6_371_000  # [m]


def burstvincentydistance(
    start_point: tuple[float, float], dist_m: float, brng: float
) -> geopy.Point:
    """
    ! returns the destination [lat/lon] point at distance dist_m [m] and at bearing brng[deg] from point pnt[lat/lon];
    default uses the geodesic distance; but also the great-circle distance can be used,
    see: https://geopy.readthedocs.io/en/stable/

    """
    d = distance(meters=dist_m)
    dest = d.destination(point=start_point, bearing=brng)
    return dest


# Source - https://stackoverflow.com/a/4913653
# Posted by Michael Dunn, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-12, License - CC BY-SA 4.0
@numba.jit
def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance in meters between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.asin(np.sqrt(a))
    r = R_EARTH
    return c * r


# Source - https://stackoverflow.com/a/54874251
# Posted by J. Taylor, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-12, License - CC BY-SA 4.0
def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2) -> float:
    geodesic = pyproj.Geod(ellps="WGS84")
    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
    return fwd_azimuth


def linspace(start: Point, stop: Point, delta: float) -> Generator[Point, None, None]:
    d_max = haversine(start.lon, start.lat, stop.lon, stop.lat)
    bearing = calculate_bearing(start.lon, start.lat, stop.lon, stop.lat)
    n = int(d_max // delta) - 1
    for d in np.linspace(0, d_max, n)[1:]:
        point = geopy.distance.distance(meters=d).destination(
            point=(start.lat, start.lon, start.alt),
            bearing=bearing,
        )
        yield Point(
            lat=point.latitude,
            lon=point.longitude,
            alt=start.alt + (stop.alt - start.alt) / d_max * d,
        )


def line_of_sight_distance(
    lat1: float, lon1: float, h1: float, lat2: float, lon2: float, h2: float
) -> float:
    """
    Calculate the line-of-sight distance between two (lat, lon, h) tuples in meters.
    """
    p1 = CoordinateTransformations.geodetic_to_cartesian(lat1, lon1, h1)
    p2 = CoordinateTransformations.geodetic_to_cartesian(lat2, lon2, h2)
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist
