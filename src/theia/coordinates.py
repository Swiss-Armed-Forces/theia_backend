import numpy as np
import pyproj
from geographiclib.geodesic import Geodesic

from theia.data_loading import elevationAt
from theia.types import Point


LATLON_BOUNDS = {
    "CH": {"lat": [45.7, 45.9], "lon": [5.7, 10.6]},
}

POSITIONS_OF_INTEREST = {
    "CH_CENTER": {"lat": 46.801111, "lon": 8.226667},
}


def sample_location(
    rng: np.random.Generator,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Point:
    lat = rng.uniform(lat_min, lat_max)
    lon = rng.uniform(lon_min, lon_max)
    alt = elevationAt(lat, lon)
    return Point(lat=lat, lon=lon, alt=alt)


class CoordinateTransformations:
    t = pyproj.Transformer.from_proj(
        pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84"),
        pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84"),
    )

    @classmethod
    def geodetic_to_cartesian(
        cls, lat: float, lon: float, alt: float
    ) -> tuple[float, float, float]:
        # the parameter sequence should be lon, lat, alt!
        x, y, z = cls.t.transform(lon, lat, alt, radians=False)
        return float(x), float(y), float(z)

    @classmethod
    def cartesian_to_geodetic(
        cls,
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float, float]:
        """Convert (x, y, z) to (lat, lon, alt)."""
        lon, lat, alt = cls.t.transform(x, y, z, radians=False, direction="INVERSE")
        return float(lat), float(lon), float(alt)


# Taken from openBURST.
def get_azimuth_between_locs(lat1, lon1, lat2, lon2):
    """

    returns azimuth (clockwise from north between point1 and point2 given in lat lon) in radians
    The shortest path between two points on the ellipsoid at (lat1, lon1) and (lat2, lon2) is called the geodesic.
    Its length is s12 and the geodesic from point 1 to point 2 has azimuths azi1 and azi2 at the two end points.
    (The azimuth is the heading measured clockwise from north. azi2 is the "forward" azimuth, i.e.,
    the heading that takes you beyond point 2 not back to point 1.)

    Parameters
    ----------
    lat1, lon1 : source position
    lat2, lon2 : destination position

    Returns
    -------
    : tmp : azimuth in radians in (0, 2pi)

    References
    ----------
    https://geographiclib.sourceforge.io/2009-03/geodesic.html

    """
    # Inverse() returns angles in (-180, 180), but we need (0, 360).
    tmp = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)["azi1"]
    tmp = (tmp + 360) % 360
    return np.radians(tmp)
