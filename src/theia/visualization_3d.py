from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

import theia.coordinates
from theia.grids import LatLonTerrainGrid
from theia.terrain_fast_los import HbvTree, _depth_to_num_nodes, _num_nodes_to_depth
from theia.types import Point


def visualize_hbv_tree(
    grid: LatLonTerrainGrid,
    tree: HbvTree,
    depth: int,
    los_points: Optional[tuple[Point, Point]] = None,
    vertical_exaggeration: float = 1.0,
) -> go.Figure:
    """
    Visualise the terrain and a level of the HBV tree in ENU coordinates.

    Parameters
    ----------
    grid: LatLonTerrainGrid
        Defines the ROI extent, resolution, and terrain model.
    tree: HbvTree
        Hierarchical bounding volume tree.  Its ``transformer`` defines the
        ENU frame used for all coordinates in the figure.
    depth: int
        Tree level to visualise (1 = root, increases toward leaves).
        Clamped to the actual tree depth when out of range.
    los_points: Optional[tuple[Point, Point]], default None
        Optional pair of geodetic points to render as a line-of-sight.
    vertical_exaggeration: float, default 1.0
        Visual stretch factor for the Up axis.  A value of 2 makes 1 m
        vertically occupy twice as much canvas space as 1 m horizontally.
        Axis tick labels are not affected — only the visual proportions change.

    Returns
    -------
    go.Figure
        Interactive plotly figure.
    """
    transformer = tree.transformer
    fig = go.Figure()

    # ------------------------------------------------------------------ #
    # Terrain mesh
    # ------------------------------------------------------------------ #
    lats = grid.latitude_values
    lons = grid.longitude_values
    n_lat, n_lon = len(lats), len(lons)

    ecef_pts = [
        theia.coordinates.CoordinateTransformations.geodetic_to_cartesian(
            lat, lon, grid.terrain_model.elevationAt(lat, lon)
        )
        for lat in lats
        for lon in lons
    ]
    enu = transformer.ecef_to_enu_multiple(ecef_pts).reshape(n_lat, n_lon, 3)

    fig.add_trace(
        go.Surface(
            x=enu[:, :, 0],
            y=enu[:, :, 1],
            z=enu[:, :, 2],
            colorscale="Greens",
            reversescale=True,
            opacity=1.0,
            showscale=False,
            name="Terrain",
            hovertemplate="E: %{x:.0f} m<br>N: %{y:.0f} m<br>U: %{z:.0f} m<extra>Terrain</extra>",
        )
    )

    # ------------------------------------------------------------------ #
    # ROI bounding box in ENU (used to cull tree nodes)
    # ------------------------------------------------------------------ #
    roi_ecef = [
        theia.coordinates.CoordinateTransformations.geodetic_to_cartesian(
            lat, lon, grid.terrain_model.elevationAt(lat, lon)
        )
        for lat, lon in [
            (grid.lat_start, grid.lon_start),
            (grid.lat_start, grid.lon_stop),
            (grid.lat_stop, grid.lon_start),
            (grid.lat_stop, grid.lon_stop),
        ]
    ]
    roi_enu = transformer.ecef_to_enu_multiple(roi_ecef)
    roi_xmin, roi_xmax = roi_enu[:, 0].min(), roi_enu[:, 0].max()
    roi_ymin, roi_ymax = roi_enu[:, 1].min(), roi_enu[:, 1].max()

    # ------------------------------------------------------------------ #
    # Tree nodes at the requested depth
    # ------------------------------------------------------------------ #
    max_depth = _num_nodes_to_depth(tree._data.shape[0])
    actual_depth = int(np.clip(depth, 1, max_depth))

    node_start = _depth_to_num_nodes(actual_depth - 1)
    node_end = min(_depth_to_num_nodes(actual_depth), tree._data.shape[0])
    nodes = tree._data[node_start:node_end]  # (M, 6) in ENU space

    # Keep only nodes whose XY footprint overlaps the ROI
    overlap_mask = (
        (nodes[:, 0] < roi_xmax)
        & (nodes[:, 3] > roi_xmin)
        & (nodes[:, 1] < roi_ymax)
        & (nodes[:, 4] > roi_ymin)
    )
    visible = nodes[overlap_mask]

    if len(visible) > 0:
        fig.add_trace(_bbox_faces(visible, actual_depth))
        fig.add_trace(_bbox_edges(visible, actual_depth))

    # ------------------------------------------------------------------ #
    # Optional line-of-sight
    # ------------------------------------------------------------------ #
    if los_points is not None:
        p1, p2 = los_points

        def _to_enu(p: Point) -> tuple[float, float, float]:
            ecef = theia.coordinates.CoordinateTransformations.geodetic_to_cartesian(
                p.lat, p.lon, p.alt
            )
            return transformer.ecef_to_enu(ecef)

        e1, e2 = _to_enu(p1), _to_enu(p2)
        fig.add_trace(
            go.Scatter3d(
                x=[e1[0], e2[0]],
                y=[e1[1], e2[1]],
                z=[e1[2], e2[2]],
                mode="lines+markers",
                line=dict(color="red", width=5),
                marker=dict(size=5, color="red"),
                name="LOS",
            )
        )

    dx = float(enu[:, :, 0].max() - enu[:, :, 0].min())
    dy = float(enu[:, :, 1].max() - enu[:, :, 1].min())
    dz = float(enu[:, :, 2].max() - enu[:, :, 2].min())
    # Normalise so the longest horizontal axis = 1; aspectratio must be O(1)
    # for Plotly's camera placement to work correctly.
    h = max(dx, dy)
    ar_x, ar_y, ar_z = dx / h, dy / h, (dz / h) * vertical_exaggeration

    fig.update_layout(
        scene=dict(
            xaxis_title="East [m]",
            yaxis_title="North [m]",
            zaxis_title="Up [m]",
            aspectmode="manual",
            aspectratio=dict(x=ar_x, y=ar_y, z=ar_z),
        ),
        title=f"HBV tree – depth {actual_depth} / {max_depth} "
        f"({len(visible)} visible nodes)",
        legend=dict(itemsizing="constant"),
    )

    return fig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_BOX_TRIS = np.array(
    [
        [0, 1, 2], [1, 3, 2],  # bottom  (z = zmin)
        [4, 5, 6], [5, 7, 6],  # top     (z = zmax)
        [0, 1, 4], [1, 5, 4],  # front   (y = ymin)
        [2, 3, 6], [3, 7, 6],  # back    (y = ymax)
        [0, 2, 4], [2, 6, 4],  # left    (x = xmin)
        [1, 3, 5], [3, 7, 5],  # right   (x = xmax)
    ],
    dtype=np.int64,
)

_BOX_EDGE_PAIRS = np.array(
    [
        [0, 1], [1, 3], [3, 2], [2, 0],  # bottom loop
        [4, 5], [5, 7], [7, 6], [6, 4],  # top loop
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ],
    dtype=np.int64,
)


def _box_corners(bbox: np.ndarray) -> np.ndarray:
    """Return the 8 corners of a box given [xmin,ymin,zmin,xmax,ymax,zmax]."""
    x0, y0, z0, x1, y1, z1 = bbox
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ]
    )


def _bbox_faces(nodes: np.ndarray, depth: int) -> go.Mesh3d:
    """Build a single semi-transparent Mesh3d for all bounding boxes."""
    n = len(nodes)
    verts = np.concatenate([_box_corners(b) for b in nodes], axis=0)  # (8n, 3)
    offsets = np.arange(n, dtype=np.int64)[:, None] * 8            # (n, 1)
    tris = (_BOX_TRIS[None] + offsets[:, None]).reshape(-1, 3)      # (12n, 3)

    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=tris[:, 0],
        j=tris[:, 1],
        k=tris[:, 2],
        color="royalblue",
        opacity=0.12,
        name=f"BVH depth {depth} (faces)",
        showlegend=True,
        showscale=False,
        hoverinfo="skip",
    )


def _bbox_edges(nodes: np.ndarray, depth: int) -> go.Scatter3d:
    """Build a single Scatter3d wireframe for all bounding boxes."""
    xs, ys, zs = [], [], []
    for bbox in nodes:
        corners = _box_corners(bbox)
        for a, b in _BOX_EDGE_PAIRS:
            xs += [corners[a, 0], corners[b, 0], None]
            ys += [corners[a, 1], corners[b, 1], None]
            zs += [corners[a, 2], corners[b, 2], None]

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="royalblue", width=1),
        name=f"BVH depth {depth} (edges)",
        showlegend=True,
        hoverinfo="skip",
    )
