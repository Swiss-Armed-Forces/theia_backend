"""
Export of positions of interest and ranges to Kadas Albireo.

The export format is a GeoPackage (``.gpkg``), which Kadas opens as vector layers
(drag & drop, or *Add layer*). A GeoPackage table holds a single geometry type, so a
:class:`KadasLayer` is written as up to two tables:

* ``<layer_name>``: one point per marker (including the centre of every range).
* ``<layer_name>_ranges``: one geodesic circle polygon per marker with radius.

Coordinates are written unchanged as WGS84 longitude/latitude, the same system as
:class:`~theia.types.Point`.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import shapely

from theia.types import Point

_CRS = "EPSG:4326"
_GEOD = pyproj.Geod(ellps="WGS84")


def _geodesic_circle(
    center: Point,
    radius: float,
    n_vertices: int = 180,
) -> shapely.Polygon:
    """Polygon approximating all points at geodesic distance ``radius`` [m] from ``center``."""
    azimuths = np.linspace(0.0, 360.0, n_vertices, endpoint=False)
    lons, lats, _ = _GEOD.fwd(
        np.full_like(azimuths, center.lon),
        np.full_like(azimuths, center.lat),
        azimuths,
        np.full_like(azimuths, radius),
    )
    return shapely.Polygon(zip(lons, lats))


class KadasLayer:
    """A named collection of markers (with optional radius) exportable to Kadas Albireo."""

    def __init__(self, layer_name: str):
        if not layer_name:
            raise ValueError("layer_name must not be empty")
        self.layer_name = layer_name
        self._markers: list[dict] = []
        self._ranges: list[dict] = []

    def add_marker(self, point: Point):
        """Add a position of interest."""
        self._markers.append(self._point_record(point, radius=None))

    def add_marker_with_radius(self, point: Point, radius: float):
        """Add a position of interest with a range circle of ``radius`` [m] around it."""
        if not radius > 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self._markers.append(self._point_record(point, radius=radius))
        self._ranges.append(
            {
                "lat": point.lat,
                "lon": point.lon,
                "alt": point.alt,
                "radius_m": radius,
                "geometry": _geodesic_circle(point, radius),
            }
        )

    def export_layer(self, output_path: str):
        """
        Write the layer to the GeoPackage ``output_path``.

        If the file already exists, only this layer's tables are replaced, so several
        layers can be exported into the same file.
        """
        if not self._markers:
            raise ValueError(f"Layer '{self.layer_name}' is empty, nothing to export")
        path = Path(output_path)
        if path.suffix.lower() != ".gpkg":
            raise ValueError(f"output_path must end in .gpkg, got '{output_path}'")

        gpd.GeoDataFrame(self._markers, crs=_CRS).to_file(
            path, layer=self.layer_name, driver="GPKG"
        )
        if self._ranges:
            gpd.GeoDataFrame(self._ranges, crs=_CRS).to_file(
                path, layer=f"{self.layer_name}_ranges", driver="GPKG"
            )

    @staticmethod
    def _point_record(point: Point, radius: float | None) -> dict:
        return {
            "lat": point.lat,
            "lon": point.lon,
            "alt": point.alt,
            "radius_m": radius,
            "geometry": shapely.Point(point.lon, point.lat),
        }
