import dash_leaflet as dl
import numpy as np
from dash import Dash, dcc, html
from flask_caching import Cache

from theia.coordinates import CoordinateTransformations
from theia.grids import LatLonHeightGrid
from theia.simulation.analysis import Analysis, death_map_to_grid
from theia.simulation.scenario_import import ScenarioFactory
from theia.simulation.theia_logging import LogLoader

skull_svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640"><!--!Font Awesome Free v7.3.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2026 Fonticons, Inc.--><path d="M480 491.4C538.5 447.4 576 379.8 576 304C576 171.5 461.4 64 320 64C178.6 64 64 171.5 64 304C64 379.8 101.5 447.4 160 491.4L160 528C160 554.5 181.5 576 208 576L240 576L240 536C240 522.7 250.7 512 264 512C277.3 512 288 522.7 288 536L288 576L352 576L352 536C352 522.7 362.7 512 376 512C389.3 512 400 522.7 400 536L400 576L432 576C458.5 576 480 554.5 480 528L480 491.4zM160 320C160 284.7 188.7 256 224 256C259.3 256 288 284.7 288 320C288 355.3 259.3 384 224 384C188.7 384 160 355.3 160 320zM416 256C451.3 256 480 284.7 480 320C480 355.3 451.3 384 416 384C380.7 384 352 355.3 352 320C352 284.7 380.7 256 416 256z"/></svg>
"""
blue_gbad = """
<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" viewBox="21 46 158 108"><path d="M25,50 l150,0 0,100 -150,0 z" stroke-width="4" stroke="black" fill="rgb(128,224,255)" fill-opacity="1" ></path><path d="M25,150 C25,110 175,110 175,150" stroke-width="4" stroke="black" fill="none" ></path></svg>
"""

skull_icon = dict(
    html=skull_svg,
    className="",  # avoid leaflet's default div-icon styling/background
    iconSize=[28, 28],
    iconAnchor=[14, 14],
)


def min_detectable_rcs_color(value, vmin, vmax):
    """
    Default light-red -> dark-red scale.
    Low values map to light red, high values map to dark red.
    """
    if vmax > vmin:
        t = (value - vmin) / (vmax - vmin)
    else:
        t = 0.0
    t = min(max(t, 0.0), 1.0)  # clamp to [0, 1]

    # Interpolate from light red (255, 200, 200) to dark red (139, 0, 0)
    light = (255, 200, 200)
    dark = (139, 0, 0)
    r = round(light[0] + (dark[0] - light[0]) * t)
    g = round(light[1] + (dark[1] - light[1]) * t)
    b = round(light[2] + (dark[2] - light[2]) * t)
    return f"rgb({r},{g},{b})"


def n_kills_overlay(
    grid, geometry: LatLonHeightGrid, vmin=None, vmax=None
) -> list[dl.Rectangle]:
    """
    Build a list of dl.Rectangle cells from a 3D grid, mirroring the
    MinDetectableRcsOverlay React component.

    grid: list/array of shape (n_i, n_j, >=1); grid[i][j][0] is the value.
    geometry: LatLonHeightGrid with lat_start, lon_start, lat_res, lon_res.
    vmin/vmax: optional fixed color-scale bounds. If omitted, they are
        computed from the non-NaN values in `grid`.
    """
    arr = np.asarray(grid, dtype=float)
    values = arr[:, :, 0]

    if vmin is None or vmax is None:
        finite_vals = values[~np.isnan(values)]
        if finite_vals.size == 0:
            vmin = vmin if vmin is not None else 0.0
            vmax = vmax if vmax is not None else 1.0
        else:
            vmin = vmin if vmin is not None else float(finite_vals.min())
            vmax = vmax if vmax is not None else float(finite_vals.max())

    cells = []
    n_i, n_j = values.shape
    for i in range(n_i):
        for j in range(n_j):
            value = values[i, j]
            if np.isnan(value):
                # No detection possible in this cell: leave it fully
                # transparent instead of forcing it onto the color scale.
                continue

            lat_lo = geometry.lat_start + (i - 0.5) * geometry.lat_res
            lon_lo = geometry.lon_start + (j - 0.5) * geometry.lon_res
            color = min_detectable_rcs_color(value, vmin, vmax)

            cells.append(
                dl.Rectangle(
                    bounds=[
                        [lat_lo, lon_lo],
                        [lat_lo + geometry.lat_res, lon_lo + geometry.lon_res],
                    ],
                    pathOptions={
                        "stroke": True,
                        "color": "black",
                        "weight": 0.1,
                        "fillColor": color,
                        "fillOpacity": 0.6,
                    },
                    children=dl.Tooltip(value),
                )
            )

    return cells


def build_map(analysis: Analysis, skull_svg: str) -> dl.Map:
    effector_markers = []
    for e in scenario.blue_orbat.gbads:
        n = analysis.n_shots_per_effector[e.gbad.id]
        color = "rgb(128, 224, 255)" if n > 0 else "gray"
        position = (e.gbad.point.lat, e.gbad.point.lon)
        m = dl.DivMarker(
            position=position,
            iconOptions=dict(
                html=blue_gbad.replace('fill="rgb(128,224,255)"', f'fill="{color}"'),
                className="",
                iconSize=[28, 28],
                iconAnchor=[14, 28],
            ),
            children=dl.Tooltip(
                html.Div(
                    [
                        f"Effector #{e.gbad.id}",
                        html.Br(),
                        f"No. shots: {n}",
                    ]
                )
            ),
        )
        effector_markers.append(m)
        effector_markers.append(
            dl.Circle(
                center=position,
                radius=e.gbad.combat_range,
                pathOptions=dict(
                    fill=False,
                    color="blue",
                    dashArray="4, 4",
                ),
            )
        )

    # RED aircraft trajectories.
    red_trajectories = []
    for target_id, t in loader.red_target_ground_truth.items():
        points = [
            CoordinateTransformations.cartesian_to_geodetic(
                *state.state_vector[[0, 2, 4]]
            )[:2]
            for state in t.states
        ]
        red_trajectories.append(
            dl.Polyline(
                positions=points,
                children=[dl.Tooltip(f"Target ID #{target_id}")],
                color="gray" if target_id in analysis.times_of_death else "red",
            )
        )

    return dl.Map(
        [
            dl.TileLayer(),
            # dl.LayerGroup(n_kills_overlay(values, grid)),
            dl.ScaleControl(position="bottomleft"),
            *[
                dl.DivMarker(
                    position=(p.lat, p.lon),
                    iconOptions=dict(
                        html=skull_svg.replace("<svg ", "<svg style='fill:red;' "),
                        # html=f'<div style="color: red;">{skull_svg}</div>',
                        className="",
                        iconSize=[28, 28],
                        iconAnchor=[14, 14],
                    ),
                    children=dl.Tooltip(
                        html.Div(
                            [
                                f"RED target #{id}",
                                html.Br(),
                                f"† {t}",
                            ]
                        )
                    ),
                )
                for id, t, p in analysis.red_death_map
            ],
            *effector_markers,
            *red_trajectories,
        ],
        center=[47.45270, 8.56068],
        zoom=10,
        style={"width": "100%", "height": "100%"},
    )


path = "/home/user/Documents/theia_backend/scenarios/fest_mw_drones/result_full_sensors.json"
scenario_path = (
    "/home/user/Documents/theia_backend/scenarios/fest_mw_drones/full_sensors.json"
)


app = Dash()
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory",
    },
)


@cache.memoize(timeout=3600)
def load_log_file(path: str) -> tuple[LogLoader, ScenarioFactory]:
    loader = LogLoader(path)
    with open(scenario_path, "r") as file:
        scenario = ScenarioFactory.model_validate_json(file.read())
    return loader, scenario


# Prepare data.
loader, scenario = load_log_file(path)
analysis = Analysis(log_file=loader)

grid, values = death_map_to_grid(analysis.red_death_map)

# GUI layout.
app.layout = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "repeat(6, 1fr)",
        "gridTemplateRows": "repeat(2, 1fr)",
        "gap": "10px",
        "height": "100vh",
        "padding": "10px",
        "boxSizing": "border-box",
    },
    children=[
        # (1,1): Graph
        html.Div(
            dcc.Graph(
                figure=analysis.plot_lifespan(is_blue=False),
                style={"width": "100%", "height": "100%"},
                config={"responsive": True},
            ),
            style={
                "gridColumn": "1 / 3",
                "gridRow": "1",
                "position": "relative",
                "minHeight": 0,
                "minWidth": 0,
            },
        ),
        # (2,1): empty placeholder
        html.Div(
            id="cell-2-1",
            style={
                "gridColumn": "1 / 3",
                "gridRow": "2",
            },
        ),
        # Columns 2-6, both rows: Map
        html.Div(
            build_map(analysis, skull_svg),
            style={
                "gridColumn": "3 / 7",
                "gridRow": "1 / 3",
            },
        ),
    ],
)


if __name__ == "__main__":
    app.run(debug=True)
