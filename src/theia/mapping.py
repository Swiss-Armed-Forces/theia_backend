import datetime

import folium
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from shapely.geometry import Polygon, LineString
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from theia.coordinates import POSITIONS_OF_INTEREST
from theia.types import Radar, Target, Trajectory


# EPSG:4326 is standard lat/lon; we reproject to 3857 (Web Mercator) for contextily tiles
WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"


class RadarMap:
    def __init__(
        self,
        radars: dict[str, Radar] = {},
        targets: dict[str, Target] = {},
        polygons: dict[str, Polygon] = {},
        paths: dict[str, LineString] = {},
        trajectories: dict[str, Trajectory] = {},
        latlon_popup: bool = True,
    ):
        self.radars = radars
        self.targets = targets
        self.polygons = polygons
        self.paths = paths
        self.trajectories = trajectories
        self.latlon_popup = latlon_popup

    def to_map(self) -> folium.folium.Map:
        map = folium.Map(
            zoom_start=10,
            location=(
                POSITIONS_OF_INTEREST["CH_CENTER"]["lat"],
                POSITIONS_OF_INTEREST["CH_CENTER"]["lon"],
            ),
            control_scale=True,
        )

        if self.latlon_popup:
            folium.LatLngPopup().add_to(map)

        for name, radar in self.radars.items():
            if radar.transmitter.point == radar.receiver.point:
                folium.Marker(
                    location=(radar.transmitter.lat, radar.transmitter.lon),
                    tooltip=f"{name} (monostatic)",
                ).add_to(map)
            else:
                folium.Marker(
                    location=(radar.transmitter.lat, radar.transmitter.lon),
                    tooltip=f"{name} (Tx)",
                ).add_to(map)
                folium.Marker(
                    location=(radar.receiver.lat, radar.receiver.lon),
                    tooltip=f"{name} (Rx)",
                ).add_to(map)

        for name, target in self.targets.items():
            folium.Marker(
                location=(target.lat, target.lon),
                tooltip=name,
                icon=folium.Icon(color="red"),
            ).add_to(map)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (name, path) in enumerate(self.paths.items()):
            folium.GeoJson(path, tooltip=name, color=colors[i % len(colors)]).add_to(
                map
            )

        for name, polygon in self.polygons.items():
            folium.GeoJson(
                polygon,
                tooltip=name,
            ).add_to(map)

        for name, trajectory in self.trajectories.items():
            folium.GeoJson(
                trajectory.to_geojson(),
                tooltip=name,
            ).add_to(map)

        return map

    def to_static_map(
        self,
        figsize: tuple[int, int] = (12, 10),
        zoom: int = 8,
        span_deg: float = 3.0,
        tile_source: cimgt.GoogleWTS = None,
    ) -> tuple[Figure, plt.Axes]:
        """
        Render the map using Cartopy with OpenStreetMap tiles.

        Args:
            figsize:    Figure dimensions in inches.
            zoom:       Tile zoom level (higher = more detail, slower).
            span_deg:   Half-width of the viewport in degrees (controls zoom-equivalent).
            tile_source: Cartopy tile object. Defaults to OSM.
                         Other options:
                           cimgt.Stamen("terrain")
                           cimgt.GoogleTiles(style="satellite")
        """
        if tile_source is None:
            tile_source = cimgt.OSM()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=tile_source.crs)

        # --- Viewport centered on CH_CENTER ---
        cx = POSITIONS_OF_INTEREST["CH_CENTER"]["lon"]
        cy = POSITIONS_OF_INTEREST["CH_CENTER"]["lat"]
        ax.set_extent(
            [cx - span_deg, cx + span_deg, cy - span_deg * 0.6, cy + span_deg * 0.6],
            crs=ccrs.PlateCarree(),
        )

        # --- Add tile basemap ---
        ax.add_image(tile_source, zoom)

        # --- Radars ---
        for name, radar in self.radars.items():
            if radar.transmitter.point == radar.receiver.point:
                ax.plot(
                    radar.transmitter.lon,
                    radar.transmitter.lat,
                    marker="^",
                    color="blue",
                    markersize=10,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
                ax.text(
                    radar.transmitter.lon,
                    radar.transmitter.lat,
                    f"  {name} (monostatic)",
                    color="blue",
                    fontsize=8,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
            else:
                ax.plot(
                    radar.transmitter.lon,
                    radar.transmitter.lat,
                    marker="^",
                    color="blue",
                    markersize=10,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
                ax.text(
                    radar.transmitter.lon,
                    radar.transmitter.lat,
                    f"  {name} (Tx)",
                    color="blue",
                    fontsize=8,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
                ax.plot(
                    radar.receiver.lon,
                    radar.receiver.lat,
                    marker="v",
                    color="cornflowerblue",
                    markersize=10,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
                ax.text(
                    radar.receiver.lon,
                    radar.receiver.lat,
                    f"  {name} (Rx)",
                    color="cornflowerblue",
                    fontsize=8,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )

        # --- Targets ---
        for name, target in self.targets.items():
            ax.plot(
                target.lon,
                target.lat,
                marker="o",
                color="red",
                markersize=10,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            ax.text(
                target.lon,
                target.lat,
                f"  {name}",
                color="red",
                fontsize=8,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

        # --- Paths ---
        for name, path in self.paths.items():
            lons, lats = path.xy
            ax.plot(
                list(lons),
                list(lats),
                color="green",
                linewidth=2,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
            mid = len(lons) // 2
            ax.text(
                lons[mid],
                lats[mid],
                name,
                color="green",
                fontsize=8,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

        # --- Polygons ---
        for name, polygon in self.polygons.items():
            lons, lats = polygon.exterior.xy
            ax.fill(
                list(lons),
                list(lats),
                color="orange",
                alpha=0.4,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            ax.plot(
                list(lons),
                list(lats),
                color="darkorange",
                linewidth=2,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            cx_p, cy_p = polygon.centroid.x, polygon.centroid.y
            ax.text(
                cx_p,
                cy_p,
                name,
                color="darkorange",
                fontsize=8,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        # --- Trajectories ---
        for name, trajectory in self.trajectories.items():
            geom = trajectory.to_geojson()
            lons, lats = geom.xy
            ax.plot(
                list(lons),
                list(lats),
                color="purple",
                linewidth=2,
                linestyle="--",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
            mid = len(lons) // 2
            ax.text(
                lons[mid],
                lats[mid],
                name,
                color="purple",
                fontsize=8,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

        # --- Legend ---
        legend_handles = []
        if self.radars:
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker="^",
                    color="blue",
                    linestyle="None",
                    markersize=8,
                    label="Radar (Tx / monostatic)",
                )
            )
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker="v",
                    color="cornflowerblue",
                    linestyle="None",
                    markersize=8,
                    label="Radar (Rx)",
                )
            )
        if self.targets:
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker="o",
                    color="red",
                    linestyle="None",
                    markersize=8,
                    label="Target",
                )
            )
        if self.paths:
            legend_handles.append(
                mlines.Line2D([], [], color="green", linewidth=2, label="Path")
            )
        if self.polygons:
            legend_handles.append(
                mpatches.Patch(
                    facecolor="orange",
                    edgecolor="darkorange",
                    alpha=0.6,
                    label="Polygon",
                )
            )
        if self.trajectories:
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="purple",
                    linewidth=2,
                    linestyle="--",
                    label="Trajectory",
                )
            )

        if legend_handles:
            ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

        fig.tight_layout()
        return fig, ax

    def to_plotly_map(
        self,
        map_style: str = "open-street-map",
        zoom: int = 6,
    ) -> go.Figure:
        """
        Render the map interactively using Plotly.

        Args:
            map_style: Mapbox tile style. Options (no token needed):
                    "open-street-map", "carto-positron", "carto-darkmatter",
                    "stamen-terrain", "stamen-toner", "stamen-watercolor"
            zoom:      Initial zoom level.
        """
        cx = POSITIONS_OF_INTEREST["CH_CENTER"]["lon"]
        cy = POSITIONS_OF_INTEREST["CH_CENTER"]["lat"]

        fig = go.Figure()

        # --- Radars ---
        for name, radar in self.radars.items():
            is_monostatic = radar.transmitter.point == radar.receiver.point

            fig.add_trace(
                go.Scattermapbox(
                    lat=[radar.transmitter.lat],
                    lon=[radar.transmitter.lon],
                    mode="markers+text",
                    marker=dict(size=12, color="blue"),
                    text=[f"{name} ({'monostatic' if is_monostatic else 'Tx'})"],
                    textposition="top right",
                    name=f"{name} (Tx/monostatic)",
                    legendgroup="radar_tx",
                    showlegend=True,
                )
            )

            if not is_monostatic:
                fig.add_trace(
                    go.Scattermapbox(
                        lat=[radar.receiver.lat],
                        lon=[radar.receiver.lon],
                        mode="markers+text",
                        marker=dict(
                            size=12, color="cornflowerblue", symbol="triangle-down"
                        ),
                        text=[f"{name} (Rx)"],
                        textposition="top right",
                        name=f"{name} (Rx)",
                        legendgroup="radar_rx",
                        showlegend=True,
                    )
                )

        # --- Targets ---
        for name, target in self.targets.items():
            fig.add_trace(
                go.Scattermapbox(
                    lat=[target.lat],
                    lon=[target.lon],
                    mode="markers+text",
                    marker=dict(size=12, color="red"),
                    text=[name],
                    textposition="top right",
                    name=name,
                    legendgroup="targets",
                    showlegend=True,
                    customdata=[
                        [
                            f"{target.lat:.4f}",
                            f"{target.lon:.4f}",
                            f"{target.alt:.0f}",  # remove if no altitude
                        ]
                    ],
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "Lat: %{customdata[0]}°<br>"
                        "Lon: %{customdata[1]}°<br>"
                        "Alt: %{customdata[2]} m<br>"
                        "<extra></extra>"
                    ),
                )
            )

        # --- Paths ---
        for name, path in self.paths.items():
            lons, lats = path.xy
            fig.add_trace(
                go.Scattermapbox(
                    lat=list(lats),
                    lon=list(lons),
                    mode="lines",
                    line=dict(width=2, color="green"),
                    name=name,
                    legendgroup="paths",
                    showlegend=True,
                )
            )

        # --- Polygons ---
        for name, polygon in self.polygons.items():
            lons, lats = polygon.exterior.xy
            fig.add_trace(
                go.Scattermapbox(
                    lat=list(lats),
                    lon=list(lons),
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,165,0,0.4)",
                    line=dict(width=2, color="darkorange"),
                    name=name,
                    legendgroup="polygons",
                    showlegend=True,
                )
            )

        # --- Trajectories ---
        for name, trajectory in self.trajectories.items():
            geom = trajectory.to_geojson()
            lons, lats = geom.xy
            fig.add_trace(
                go.Scattermapbox(
                    lat=list(lats),
                    lon=list(lons),
                    mode="lines",
                    line=dict(
                        width=2, color="purple", dash="dot"
                    ),  # "dot" ≈ dashed in mapbox
                    name=name,
                    legendgroup="trajectories",
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{name}</b><br>Lat: %{{lat:.4f}}°<br>Lon: %{{lon:.4f}}°<br>"
                    ),
                )
            )

        fig.update_layout(
            mapbox=dict(
                style=map_style,
                center=dict(lat=cy, lon=cx),
                zoom=zoom,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
            ),
        )

        return fig


def plot_trajectories(
    times: list[datetime.datetime],
    radars: dict[str, Radar],
    trajectories: list[Trajectory],
):
    frames = []
    for i, time in enumerate(times):
        targets = [trajectory(time) for trajectory in trajectories]
        targets = [target for target in targets if target is not None]
        fig = RadarMap(
            radars=radars,
            targets={str(t.id): t for t in targets},
        ).to_plotly_map()
        frames.append(
            go.Frame(
                data=fig.data,
                name=str(time),
                layout=go.Layout(title_text=f"Time: {time}"),
            )
        )

    # --- Initial (first) frame data ---
    base_fig = RadarMap(
        radars=radars,
        targets={
            str(t.id): t for t in targets
        },  # or reuse frame_fig from last iteration
    ).to_plotly_map()

    fig = go.Figure(
        data=frames[0].data,
        layout=base_fig.layout,  # <-- this brings mapbox config along
        frames=frames,
    )

    # --- Slider steps ---
    slider_steps = [
        dict(
            args=[
                [f.name],
                {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            label=str(t),
            method="animate",
        )
        for f, t in zip(frames, times)
    ]

    # --- Layout: only add animation controls, don't touch mapbox or axes ---
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=-0.15,
                x=0.05,
                xanchor="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Stop",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Time: ", visible=True, xanchor="right"),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
    )
    return fig
