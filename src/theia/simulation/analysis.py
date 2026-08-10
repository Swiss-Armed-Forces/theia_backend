import datetime

import numpy as np
import plotly.graph_objects as go

from theia.grids import LatLonHeightGrid
from theia.types import Point


def plot_lifespan(
    time_of_birth: dict[int, datetime.datetime],
    time_of_death: dict[int, datetime.datetime],
    t_max: datetime.datetime,
):
    red_target_ids = sorted(
        list(time_of_birth.keys()),
        key=lambda id: time_of_birth[id],
    )

    forest_green = "rgb(46, 111, 64)"
    coral = "rgb(248, 131, 121)"

    fig = go.Figure()

    for i, id in enumerate(red_target_ids):
        start = time_of_birth[id]
        end = time_of_death.get(id, t_max)

        start = time_of_birth[id]
        end = time_of_death.get(id, t_max)

        is_killed = id in time_of_death

        xs = [
            datetime.datetime.fromtimestamp(t, datetime.UTC)
            for t in np.arange(start.timestamp(), end.timestamp(), 10)
        ]
        ys = [i] * len(xs)

        # lifeline
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=coral if is_killed else forest_green, width=3),
                showlegend=False,
                hovertemplate=f"Target ID {id}<extra></extra>",
            )
        )

        # skull marker at time of death, centered on the line
        if is_killed:
            fig.add_trace(
                go.Scatter(
                    x=[time_of_death[id]],
                    y=[i],
                    mode="text",
                    text=["☠"],
                    textfont=dict(size=20),
                    textposition="middle center",
                    showlegend=False,
                )
            )

    fig.update_layout(
        # width=800,
        # height=int(2 * 4.5 * 100),  # match matplotlib figsize scaling
        # yaxis=dict(
        #     tickmode="array",
        #     tickvals=red_target_ids,
        #     title="Target ID",
        # ),
        margin=dict(l=30, r=10, t=30, b=30),  # left, right, top, bottom in pixels
        autosize=True,
        xaxis=dict(title="Time"),
        # template="plotly_white",
        title=dict(text="RED target lifetimes", xanchor="center"),
    )

    return fig


def death_map_to_grid(
    death_map: list[tuple[datetime.datetime, Point]],
) -> tuple[LatLonHeightGrid, np.ndarray]:
    values, xs, ys = np.histogram2d(
        [p.lat for id, t, p in death_map],
        [p.lon for id, t, p in death_map],
        bins=15,
    )
    grid = LatLonHeightGrid(
        lat_start=xs[0],
        lat_stop=xs[-1],
        lat_res=xs[1] - xs[0],
        lon_start=ys[0],
        lon_stop=ys[-1],
        lon_res=ys[1] - ys[0],
        height_start=0,
        height_stop=0,
        height_res=1,
    )
    values = values.reshape((values.shape[0], values.shape[1], 1))

    return grid, values
