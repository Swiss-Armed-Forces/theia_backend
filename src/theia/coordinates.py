import numpy as np
import pyproj
from geographiclib.geodesic import Geodesic


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
        return [x, y, z]


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
    : tmp : azimuth in degrees

    References
    ----------
    https://geographiclib.sourceforge.io/2009-03/geodesic.html

    """
    tmp = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    return np.radians(tmp["azi1"])
