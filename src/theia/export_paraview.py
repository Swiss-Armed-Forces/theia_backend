import itertools
from typing import Callable, Literal

import numpy as np
import struct
import base64
from pathlib import Path

import pandas as pd
import pydantic

from theia.coordinates import CoordinateTransformations, _ecef_to_enu_rotation_matrix
from theia.terrain import elevationAt


class PointOfInterest(pydantic.BaseModel):
    id: int
    label: str
    type: Literal["Rx", "Tx"]
    lat: float
    lon: float
    alt: float


def _encode_array(arr: np.ndarray) -> str:
    """Base64-encode a numpy array with a 4-byte length header (VTK appended format)."""
    raw = arr.tobytes()
    header = struct.pack("<I", len(raw))  # uint32 little-endian byte count
    return base64.b64encode(header + raw).decode("ascii")


def write_ecef_structured_grid(
    ecef_points: np.ndarray,  # (N, 3) float64, ECEF metres
    n_lat: int,
    n_lon: int,
    scalars: dict[str, np.ndarray],  # name -> (N,) float64
    filepath: str | Path,
):
    """
    Write a VTK StructuredGrid (.vts) from ECEF points.
    n_lat = number of latitude samples  (slow index, axis 0)
    n_lon = number of longitude samples (fast index, axis 1)
    Points must be in row-major order: lat outer, lon inner.
    """
    assert ecef_points.shape == (n_lat * n_lon, 3)
    filepath = Path(filepath).with_suffix(".vts")

    N = n_lat * n_lon

    # VTK structured grid dimensions: (ni, nj, nk)
    # We lay out as (n_lon, n_lat, 1) — lon is i (fast), lat is j (slow)
    ni, nj, nk = n_lon, n_lat, 1

    # VTK expects points interleaved as x0,y0,z0, x1,y1,z1, ...
    # Our array is already (N, 3) row-major — just ensure float64 C-contiguous
    points_vtk = np.ascontiguousarray(ecef_points, dtype=np.float64)  # (N, 3)

    def enc(arr):
        return _encode_array(np.ascontiguousarray(arr, dtype=np.float64))

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append(
        f'<VTKFile type="StructuredGrid" version="0.1" '
        f'byte_order="LittleEndian" header_type="UInt32">'
    )
    lines.append(f'  <StructuredGrid WholeExtent="0 {ni - 1} 0 {nj - 1} 0 {nk - 1}">')
    lines.append(f'    <Piece Extent="0 {ni - 1} 0 {nj - 1} 0 {nk - 1}">')

    # ---- Point coordinates ----
    lines.append("      <Points>")
    lines.append(
        '        <DataArray type="Float64" NumberOfComponents="3" '
        f'format="binary">{enc(points_vtk)}</DataArray>'
    )
    lines.append("      </Points>")

    # ---- Point data (scalars) ----
    if scalars:
        lines.append(f'      <PointData Scalars="{next(iter(scalars))}">')
        for name, arr in scalars.items():
            assert arr.shape == (N,), (
                f"Scalar '{name}': expected ({N},), got {arr.shape}"
            )
            lines.append(
                f'        <DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="1" format="binary">{enc(arr)}</DataArray>'
            )
        lines.append("      </PointData>")

    lines.append("    </Piece>")
    lines.append("  </StructuredGrid>")
    lines.append("</VTKFile>")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {filepath}  ({filepath.stat().st_size / 1e6:.1f} MB)")


class ParaviewExporter:
    def __init__(
        self,
        lat_min: float,
        lat_max: float,
        lat_res: float,
        lon_min: float,
        lon_max: float,
        lon_res: float,
        elevation_factor: float = 10.0,
    ):
        self._lat_min = lat_min
        self._lat_max = lat_max
        self._lat_res = lat_res
        self._lon_min = lon_min
        self._lon_max = lon_max
        self._lon_res = lon_res
        self._elevation_factor = elevation_factor

        self._lats = np.arange(self._lat_min, self._lat_max, self._lat_res)
        self._lons = np.arange(self._lon_min, self._lon_max, self._lon_res)
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        self._center = np.array(
            (
                center_lat,
                center_lon,
                elevationAt(center_lat, center_lon),
            )
        )
        self._rotation = _ecef_to_enu_rotation_matrix(self._center[0], self._center[1])

    def _build_points_geodetic(self) -> np.ndarray:
        points = np.empty((len(self._lats) * len(self._lons), 3), dtype=np.float32)
        i = 0
        for j, lat in enumerate(self._lats):
            for k, lon in enumerate(self._lons):
                alt = elevationAt(lat, lon)
                if alt < 0:
                    # Missing values in the dataset. We fill with the closest non-missing.
                    r = 1
                    while alt < 0:
                        patch = list(
                            itertools.product(
                                self._lats[j - r : j + r],
                                self._lons[k - r : k + r],
                            )
                        )
                        patch_alts = [elevationAt(lat, lon) for lat, lon in patch]
                        alt = np.max(patch_alts)
                        r += 1

                points[i, :] = (
                    lat,
                    lon,
                    np.clip(self._elevation_factor * alt, 0, np.inf),
                )
                i += 1
        return points

    def _build_points_ecef(self) -> np.ndarray:
        points = self._build_points_geodetic()
        for i, point in enumerate(points):
            points[i, :] = CoordinateTransformations.geodetic_to_cartesian(*point)
        return points

    def export_terrain(
        self,
        path: str,
        functions: dict[str, Callable[[float, float, float], float]] = {},
    ):
        points = self._build_points_ecef()
        points_transformed = points - self._center
        points_transformed = (self._rotation @ points_transformed.T).T

        scalars: dict[str, list[float]] = {}
        for name, fn in functions.items():
            scalars[name] = np.array([fn(p[0], p[1], p[2]) for p in points]).flatten()

        write_ecef_structured_grid(
            points_transformed,
            len(self._lats),
            len(self._lons),
            scalars,
            path,
        )

    def export_pois(self, pois: list[PointOfInterest], path: str):
        df = []
        for poi in pois:
            x, y, z = CoordinateTransformations.geodetic_to_cartesian(
                poi.lat,
                poi.lon,
                poi.alt * self._elevation_factor,
            )
            x, y, z = self._rotation @ np.array(
                (
                    x - self._center[0],
                    y - self._center[1],
                    z - self._center[2],
                )
            )
            df.append(
                {
                    "ID": poi.id,
                    "label": poi.label,
                    "type": poi.type,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
        df = pd.DataFrame(df).drop_duplicates()
        df.to_csv(path, index=False)
