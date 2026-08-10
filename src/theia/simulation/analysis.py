from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime

import numpy as np
import plotly.graph_objects as go

from theia.grids import LatLonHeightGrid
from theia.simulation.theia_logging import LogLoader
from theia.types import Point


@dataclass
class Analysis:
    log_file: LogLoader

    def __post_init__(self):
        red_death_map: list[tuple[int, datetime.datetime, Point]] = []
        blue_death_map: list[tuple[int, datetime.datetime, Point]] = []
        for event in self.log_file.kill_events:
            snapshot = next(
                (s for s in self.log_file._snapshots if s.time == event.time)
            )
            target = next(
                (t for t in snapshot.red_targets if t.id == event.target_id),
                None,
            )
            if target is not None:
                red_death_map.append((event.target_id, event.time, target.point))
                continue
            target = next(
                (t for t in snapshot.blue_targets if t.id == event.target_id),
                None,
            )
            if target is not None:
                blue_death_map.append((event.target_id, event.time, target.point))
                continue
            raise RuntimeError(
                "What exactly dies here? Target is neither blue nor red...",
                event,
            )
        self._red_death_map = red_death_map
        self._blue_death_map = blue_death_map

    def times_of_birth(self, is_blue: bool) -> dict[int, datetime.datetime]:
        if is_blue:
            raise NotImplementedError()
        ground_truth = self.log_file.red_target_ground_truth
        return {
            target_id: trajectory.states[0].timestamp
            for target_id, trajectory in ground_truth.items()
        }

    @property
    def times_of_death(self) -> dict[int, datetime.datetime]:
        return {e.target_id: e.time for e in self.log_file.kill_events}

    @property
    def n_shots_per_effector(self) -> defaultdict[int, int]:
        return defaultdict(lambda: 0) | dict(
            Counter([shot.shooter.id for shot in self.log_file.shots])
        )

    @property
    def red_death_map(self) -> list[tuple[int, datetime.datetime, Point]]:
        return self._red_death_map

    @property
    def blue_death_map(self) -> list[tuple[int, datetime.datetime, Point]]:
        return self._blue_death_map

    def plot_lifespan(self, is_blue: bool):
        red_target_ids = sorted(
            list(self.times_of_birth(is_blue)),
            key=lambda id: self.times_of_birth(is_blue)[id],
        )

        forest_green = "rgb(46, 111, 64)"
        coral = "rgb(248, 131, 121)"

        fig = go.Figure()

        for i, id in enumerate(red_target_ids):
            start = self.times_of_birth(is_blue)[id]
            end = self.times_of_death.get(id, self.log_file.t_max)

            is_killed = id in self.times_of_death

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
                        x=[self.times_of_death[id]],
                        y=[i],
                        mode="text",
                        text=["☠"],
                        textfont=dict(size=20),
                        textposition="middle center",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            # left, right, top, bottom in pixels
            margin=dict(l=30, r=10, t=30, b=30),
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
