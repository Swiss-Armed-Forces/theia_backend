import pyproj


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
