import math
from typing import Generator
import geopy
from geopy.distance import distance
import numba
import numpy as np
import pyproj

from theia.coordinates import CoordinateTransformations
from theia.types import Point


R_EARTH = 6_371_000  # [m]


def burstvincentydistance(
    start_point: tuple[float, float], dist_m: float, brng: float, alt: float
) -> Point:
    """
    ! returns the destination [lat/lon] point at distance dist_m [m] and at bearing brng[deg] from point pnt[lat/lon];
    default uses the geodesic distance; but also the great-circle distance can be used,
    see: https://geopy.readthedocs.io/en/stable/

    """
    d = distance(meters=dist_m)
    dest = d.destination(point=start_point, bearing=brng)
    return Point(lat=dest.latitude, lon=dest.longitude, alt=alt)


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
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
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
    """
    Sample points with uniform spacing (geodesic distance).

    Parameters
    ----------
    start: Point
        Start point.
    stop: Point
        End point.
    delta: float
        Spacing between points in geodesic distance [m].
    """
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

    Returns
    -------
    float
        Line-of-sight distance [m]
    """
    p1 = CoordinateTransformations.geodetic_to_cartesian(lat1, lon1, h1)
    p2 = CoordinateTransformations.geodetic_to_cartesian(lat2, lon2, h2)
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist


def get_2d_distance_between_locs_heights(
    lat1: float,
    lon1: float,
    h1: float,
    lat2: float,
    lon2: float,
    h2: float,
) -> float:
    """
    Calculate distance between two lat lons and heights using ecef cartesian transformation [km].

    Parameters
    ----------
    lat1: float
        Latitude  of point 1 [°]
    lon1: float:
        Longitude of point 1 [°]
    h1: float
        Altitude (meters above sea level) of point 1 [m].
    lat1: float
        Latitude  of point 2 [°]
    lon1: float:
        Longitude of point 2 [°]
    h1: float
        Altitude (meters above sea level) of point 2 [m].

    Returns
    -------
    float
        Distance between point 1 and point 2 along line-of-sight in Cartesian coordinates [km].
    """
    p1 = CoordinateTransformations.geodetic_to_cartesian(lat1, lon1, h1)
    p2 = CoordinateTransformations.geodetic_to_cartesian(lat2, lon2, h2)
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist / 1000.0  # [km]


def get_bistatic_range(
    tx_latlonalt: tuple[float, float, float],
    rx_latlonalt: tuple[float, float, float],
    tgt_latlonalt: tuple[float, float, float],
):
    """
    Calculate bistatic range [km],  tgt_rx_range [km], tgt_tx_range [km], baseline_range [km])
    for given Tx, Rx and Target
    input Tx: lat, lon, alt[masl] + antenna height [magl]
    input Rx: lat, lon, alt[masl] + antenna height [magl]
    input tgt: lat, lon, alt[masl]
    definition bistatic range [km]  = distance(Tx -> Tgt -> Rx ) - distance(Rx->Tx)

    Calculate bistatic range and its components [km].

    Returns
    -------
        bistatic_range: float
            Distance(Tx - target - Rx) - distance(Tx - Rx) [km]
        tgt_rx_range: float
            Line-of-sight distance(target - Rx) [km]
        tgt_tx_range: float
            Line-of-sight distance(between Tx - target) [km]
        baseline_range: float
            Line-of-sight distance(between Rx - Tx) [km]
    """

    tx_tgt_range = get_2d_distance_between_locs_heights(
        tx_latlonalt[0],
        tx_latlonalt[1],
        tx_latlonalt[2],
        tgt_latlonalt[0],
        tgt_latlonalt[1],
        tgt_latlonalt[2],
    )
    tgt_rx_range = get_2d_distance_between_locs_heights(
        rx_latlonalt[0],
        rx_latlonalt[1],
        rx_latlonalt[2],
        tgt_latlonalt[0],
        tgt_latlonalt[1],
        tgt_latlonalt[2],
    )
    tx_rx_range = get_2d_distance_between_locs_heights(
        tx_latlonalt[0],
        tx_latlonalt[1],
        tx_latlonalt[2],
        rx_latlonalt[0],
        rx_latlonalt[1],
        rx_latlonalt[2],
    )

    return (
        tx_tgt_range + tgt_rx_range - tx_rx_range,
        tgt_rx_range,
        tx_tgt_range,
        tx_rx_range,
    )
