from dataclasses import dataclass
import itertools
from typing import Callable, Literal

import numpy as np
import struct
import base64
from pathlib import Path

import pandas as pd
import pydantic

from theia.coordinates import CoordinateTransformations, _ecef_to_enu_rotation_matrix
from theia.terrain import AbstractTerrainModel
from theia.types import Trajectory


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
        '<VTKFile type="StructuredGrid" version="0.1" '
        'byte_order="LittleEndian" header_type="UInt32">'
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


@dataclass
class Line:
    """A named polyline defined by geodetic waypoints."""

    label: str
    points: list[tuple[float, float, float]]  # (lat, lon, alt) in degrees / metres


def write_ecef_polylines(
    lines: list[np.ndarray],  # each entry: (N_i, 3) float64 ECEF metres
    filepath: str | Path,
    line_labels: list[str] | None = None,
    scalars: dict[str, np.ndarray] | None = None,  # name -> (total_points,) float64
):
    """
    Write a VTK PolyData (.vtp) file containing one polyline per entry in *lines*.

    Parameters
    ----------
    lines:
        List of (N_i, 3) float64 arrays in ECEF metres (already transformed to the
        local ENU frame, exactly as *export_terrain* does with its points).
    filepath:
        Output path; the ``.vtp`` extension is enforced automatically.
    line_labels:
        Optional per-line string labels stored as CellData.
    scalars:
        Optional per-point scalar fields stored as PointData.
    """
    filepath = Path(filepath).with_suffix(".vtp")

    # ------------------------------------------------------------------ #
    # Flatten all lines into a single point array and build VTK            #
    # connectivity / offsets arrays.                                        #
    # ------------------------------------------------------------------ #
    all_points = np.concatenate(lines, axis=0).astype(np.float64)  # (Ntotal, 3)
    total_pts = len(all_points)

    # connectivity: [0, 1, …, n0-1, n0, n0+1, …]  — point indices in order
    connectivity = np.arange(total_pts, dtype=np.int64)

    # offsets: one entry per cell (polyline), value = cumulative point count
    lengths = np.array([len(ln) for ln in lines], dtype=np.int64)
    offsets = np.cumsum(lengths)  # (n_lines,)

    n_cells = len(lines)

    def enc(arr, dtype=np.float64):
        return _encode_array(np.ascontiguousarray(arr, dtype=dtype))

    xml_lines: list[str] = []
    xml_lines.append('<?xml version="1.0" encoding="utf-8"?>')
    xml_lines.append(
        '<VTKFile type="PolyData" version="0.1" '
        'byte_order="LittleEndian" header_type="UInt32">'
    )
    xml_lines.append("  <PolyData>")
    xml_lines.append(
        f'    <Piece NumberOfPoints="{total_pts}" NumberOfLines="{n_cells}" '
        f'NumberOfVerts="0" NumberOfStrips="0" NumberOfPolys="0">'
    )

    # ---- Point coordinates ------------------------------------------- #
    xml_lines.append("      <Points>")
    xml_lines.append(
        '        <DataArray type="Float64" NumberOfComponents="3" '
        f'format="binary">{enc(all_points)}</DataArray>'
    )
    xml_lines.append("      </Points>")

    # ---- Line connectivity -------------------------------------------- #
    xml_lines.append("      <Lines>")
    xml_lines.append(
        '        <DataArray type="Int64" Name="connectivity" '
        f'format="binary">{enc(connectivity, np.int64)}</DataArray>'
    )
    xml_lines.append(
        '        <DataArray type="Int64" Name="offsets" '
        f'format="binary">{enc(offsets, np.int64)}</DataArray>'
    )
    xml_lines.append("      </Lines>")

    # ---- Per-point scalars (PointData) --------------------------------- #
    if scalars:
        xml_lines.append(f'      <PointData Scalars="{next(iter(scalars))}">')
        for name, arr in scalars.items():
            assert arr.shape == (total_pts,), (
                f"Scalar '{name}': expected ({total_pts},), got {arr.shape}"
            )
            xml_lines.append(
                f'        <DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="1" format="binary">{enc(arr)}</DataArray>'
            )
        xml_lines.append("      </PointData>")

    # ---- Per-line labels (CellData) ------------------------------------ #
    if line_labels:
        assert len(line_labels) == n_cells
        # VTK stores strings as a flat char array + per-string offsets
        joined = "".join(line_labels).encode("utf-8")
        char_offsets = np.cumsum(
            [len(lbl.encode("utf-8")) for lbl in line_labels], dtype=np.int64
        )
        xml_lines.append("      <CellData>")
        xml_lines.append(
            '        <DataArray type="UInt8" Name="label_chars" '
            f'format="binary">{enc(np.frombuffer(joined, dtype=np.uint8), np.uint8)}</DataArray>'
        )
        xml_lines.append(
            '        <DataArray type="Int64" Name="label_offsets" '
            f'format="binary">{enc(char_offsets, np.int64)}</DataArray>'
        )
        xml_lines.append("      </CellData>")

    xml_lines.append("    </Piece>")
    xml_lines.append("  </PolyData>")
    xml_lines.append("</VTKFile>")

    filepath.write_text("\n".join(xml_lines), encoding="utf-8")
    print(f"Written: {filepath}  ({filepath.stat().st_size / 1e6:.2f} MB)")


class ParaviewExporter:
    def __init__(
        self,
        output_dir: str,
        lat_min: float,
        lat_max: float,
        lat_res: float,
        lon_min: float,
        lon_max: float,
        lon_res: float,
        terrain_model: AbstractTerrainModel,
        elevation_factor: float = 10.0,
        fill_negative_alts: bool = True,
    ):
        self._terrain_path = Path(f"{output_dir}/terrain.vts").absolute()
        self._pois_path = Path(f"{output_dir}/pois.csv").absolute()
        self._trajectories_path = Path(f"{output_dir}/trajectories.vtp").absolute()
        self._script_path = f"{output_dir}/script.py"
        self._lat_min = lat_min
        self._lat_max = lat_max
        self._lat_res = lat_res
        self._lon_min = lon_min
        self._lon_max = lon_max
        self._lon_res = lon_res
        self._elevation_factor = elevation_factor
        self._terrain_model = terrain_model
        self._fill_negative_alts = fill_negative_alts

        self._lats = np.arange(self._lat_min, self._lat_max, self._lat_res)
        self._lons = np.arange(self._lon_min, self._lon_max, self._lon_res)
        self._center = np.array(
            CoordinateTransformations.geodetic_to_cartesian(
                lat_min,
                lon_min,
                0,
            )
        )
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        self._rotation = _ecef_to_enu_rotation_matrix(center_lat, center_lon)

    def _build_points_geodetic(self) -> np.ndarray:
        points = np.empty((len(self._lats) * len(self._lons), 3), dtype=np.float32)
        i = 0
        for j, lat in enumerate(self._lats):
            for k, lon in enumerate(self._lons):
                alt = self._terrain_model.elevationAt(lat, lon)
                if alt < 0 and self._fill_negative_alts:
                    # Missing values in the dataset. We fill with the closest non-missing.
                    r = 1
                    while alt < 0:
                        patch = list(
                            itertools.product(
                                self._lats[j - r : j + r],
                                self._lons[k - r : k + r],
                            )
                        )
                        patch_alts = [
                            self._terrain_model.elevationAt(lat, lon)
                            for lat, lon in patch
                        ]
                        alt = np.max(patch_alts)
                        r += 1

                points[i, :] = (
                    lat,
                    lon,
                    alt,
                )
                i += 1
        return points

    def _build_points_ecef(self) -> np.ndarray:
        points = self._build_points_geodetic()
        for i, point in enumerate(points):
            points[i, :] = CoordinateTransformations.geodetic_to_cartesian(*point)
        return points

    def _export_terrain(
        self,
        path: str,
        functions: dict[str, Callable[[float, float, float], float]] = {},
    ):
        points = self._build_points_ecef()
        points_transformed = points - self._center
        points_transformed = (self._rotation @ points_transformed.T).T
        points_transformed[:, 2] *= self._elevation_factor

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

    def _export_pois(self, pois: list[PointOfInterest], path: str):
        df = []
        for poi in pois:
            x, y, z = CoordinateTransformations.geodetic_to_cartesian(
                poi.lat,
                poi.lon,
                poi.alt,
            )
            x, y, z = self._rotation @ np.array(
                (
                    x - self._center[0],
                    y - self._center[1],
                    z - self._center[2],
                )
            )
            z *= self._elevation_factor
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

    def _export_trajectories(
        self,
        trajectories: list[Trajectory],
        path: str,
        scalars: dict[str, Callable[[float, float, float], float]] | None = None,
    ):
        """
        Export a list of polylines to a VTK PolyData file (.vtp).

        Each :class:`Line` carries a label and a sequence of ``(lat, lon, alt)``
        waypoints.  The altitude is scaled and the resulting ECEF points are
        transformed to the same local ENU frame used by :meth:`export_terrain`
        and :meth:`export_pois`, so all three datasets align in ParaView without
        any extra registration step.

        Parameters
        ----------
        trajectories:
            Trajectories to export.
        path:
            Output file path (the ``.vtp`` extension is enforced automatically).
        scalars:
            Optional mapping of scalar-field name to a callable
            ``f(lat, lon, alt) -> float`` evaluated at every waypoint.  The
            *alt* value passed in is the **raw** (unscaled) altitude in metres.
        """
        lines = []
        for traj in trajectories:
            points = [
                (lat, lon, alt)
                for lat, lon, alt in zip(traj.lats, traj.lons, traj.alts, strict=True)
            ]
            lines.append(
                Line(
                    "",
                    points,
                ),
            )

        ecef_line_arrays: list[np.ndarray] = []
        all_geodetic: list[np.ndarray] = []  # kept for scalar evaluation

        for line in lines:
            pts_geo = np.array(
                [(lat, lon, alt) for lat, lon, alt in line.points],
                dtype=np.float64,
            )
            all_geodetic.append(pts_geo)

            # Scale elevation, convert to ECEF, then to local ENU frame.
            pts_ecef = np.array(
                [
                    CoordinateTransformations.geodetic_to_cartesian(lat, lon, alt)
                    for lat, lon, alt in pts_geo
                ],
                dtype=np.float64,
            )
            pts_local = (self._rotation @ (pts_ecef - self._center).T).T
            pts_local[:, 2] *= self._elevation_factor
            ecef_line_arrays.append(pts_local)

        # Build per-point scalar arrays if requested.
        vtk_scalars: dict[str, np.ndarray] | None = None
        if scalars:
            vtk_scalars = {}
            for name, fn in scalars.items():
                values = np.concatenate(
                    [
                        np.array([fn(lat, lon, alt) for lat, lon, alt in pts_geo])
                        for pts_geo in all_geodetic
                    ]
                )
                vtk_scalars[name] = values

        write_ecef_polylines(
            ecef_line_arrays,
            path,
            line_labels=[ln.label for ln in lines],
            # scalars=vtk_scalars,
        )

    def _export_paraview_script(
        self,
        terrain_scalar: str | None = None,
        terrain_colormap: str = "Cool to Warm",
        line_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        rx_color: tuple[float, float, float] = (0.2, 0.6, 1.0),
        tx_color: tuple[float, float, float] = (1.0, 0.3, 0.1),
        poi_glyph_size: float = 50.0,
    ):
        """
        Write a ParaView Python script that loads the terrain (.vts), POIs (.csv),
        and lines (.vtp) produced by the other export_* methods, and configures
        color/representation settings so the scene is ready to use immediately.

        Run it in ParaView via:
            Tools → Macro → Add Macro  (interactive)
            pvpython <script>.py        (headless / batch)

        Parameters
        ----------
        terrain_scalar:
            Name of the scalar array to colour the terrain by.  If None, the terrain
            is rendered as a plain grey surface.
        terrain_colormap:
            Any ParaView built-in preset name, e.g. "Cool to Warm", "Viridis", "Rainbow".
        line_color:
            RGB tuple (0-1) applied uniformly to all exported polylines.
        rx_color, tx_color:
            RGB tuples (0-1) for Rx and Tx POI glyphs respectively.
        poi_glyph_size:
            Sphere glyph radius (in the same units as the exported coordinates).
        """
        terrain_path = self._terrain_path
        pois_path = self._pois_path
        lines_path = self._trajectories_path
        output_script_path = self._script_path

        def rgb(t: tuple) -> str:
            return f"[{t[0]}, {t[1]}, {t[2]}]"

        terrain_color_block = ""
        if terrain_scalar:
            terrain_color_block = f"""\
ColorBy(terrain_display, ('POINTS', {terrain_scalar!r}))
terrain_lut = GetColorTransferFunction({terrain_scalar!r})
terrain_lut.ApplyPreset({terrain_colormap!r}, True)
terrain_lut.RescaleTransferFunctionToDataRange(True, False)
scalar_bar = GetScalarBar(terrain_lut, view)
scalar_bar.Title = {terrain_scalar!r}
scalar_bar.Visibility = 1
    """
        else:
            terrain_color_block = """\
terrain_display.ColorArrayName = [None, '']
terrain_display.DiffuseColor = [0.6, 0.6, 0.6]
    """

        script = f"""\
# Auto-generated ParaView visualisation script.
# Run via: Tools → Macro → Add Macro, or:  pvpython {output_script_path}
# ── generated by ParaviewExporter ────────────────────────────────────────────

from paraview.simple import (
    XMLStructuredGridReader, XMLPolyDataReader, CSVReader,
    TableToPoints, Glyph, Threshold, ExtractSelection,
    GetActiveViewOrCreate, Show, Hide, ColorBy,
    GetColorTransferFunction, GetOpacityTransferFunction,
    GetScalarBar, RenderAllViews, ResetCamera,
    CreateRenderView,
)
import paraview.simple as pvs

pvs._DisableFirstRenderCameraReset()
view = GetActiveViewOrCreate('RenderView')
# view.Background = [0.15, 0.15, 0.18]
# view.BackgroundColorMode = 0

# ── 1. Terrain ───────────────────────────────────────────────────────────────
terrain_reader = XMLStructuredGridReader(
    FileName=[r{str(terrain_path)!r}]
)
terrain_reader.UpdatePipeline()

terrain_display = Show(terrain_reader, view)
terrain_display.Representation = 'Surface'
terrain_display.Opacity = 1.0
# terrain_display.Specular = 0.1
{terrain_color_block}
# ── 2. Lines ─────────────────────────────────────────────────────────────────
lines_reader = XMLPolyDataReader(
    FileName=[r{str(lines_path)!r}]
)
lines_reader.CellArrayStatus = []
lines_reader.TimeArray = 'None'
lines_reader.UpdatePipeline()

lines_display = Show(lines_reader, view)
lines_display.Representation = 'Wireframe'
lines_display.ColorArrayName = [None, '']
lines_display.AmbientColor = {rgb(line_color)}
lines_display.DiffuseColor = {rgb(line_color)}
lines_display.LineWidth = 2.5

# ── 3. POIs ──────────────────────────────────────────────────────────────────
# The CSV carries x/y/z columns produced by export_pois().
csv_reader = CSVReader(FileName=[r{str(pois_path)!r}])
csv_reader.UpdatePipeline()

poi_points = TableToPoints(Input=csv_reader)
poi_points.XColumn = 'x'
poi_points.YColumn = 'y'
poi_points.ZColumn = 'z'
poi_points.UpdatePipeline()

poi_display = Show(poi_points, view)
poi_display.Representation = 'Points'
poi_display.PointSize = 10.0
poi_display.ColorArrayName = [None, '']
poi_display.AmbientColor = [0.0, 0.0, 1.0]
poi_display.DiffuseColor = [0.0, 0.0, 1.0]
poi_display.RenderPointsAsSpheres = True

# ── 4. Finalise ───────────────────────────────────────────────────────────────
view.ResetCamera()
RenderAllViews()
print("Scene loaded successfully.")
    """
        with open(output_script_path, "w", encoding="utf-8") as file:
            file.write(script)
            print(f"Written: {output_script_path}")

    def export(
        self,
        trajectories: list[Trajectory],
        pois: list[PointOfInterest],
    ):
        self._export_terrain(self._terrain_path)
        if len(trajectories) > 0:
            self._export_trajectories(trajectories, self._trajectories_path)
        if len(pois) > 0:
            self._export_pois(pois, self._pois_path)
        self._export_paraview_script()
