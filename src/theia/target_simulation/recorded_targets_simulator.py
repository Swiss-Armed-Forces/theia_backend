from collections.abc import Iterable
import copy
import datetime

import numpy as np
from scipy.interpolate import CubicSpline

from theia.types import Point, Target, TargetSimulator, Trajectory


class RecordedTargetsSimulator(TargetSimulator):
    """
    Simulate targets based on pre-recorded trajectories.

    Trajectories are interpolated using cubic splines in time.
    Positions are in angles (lat, lon) and meters above sea level (alt).
    Velocities are interpolated in meters per second along longitude,
    latitude and radial direction. Therefore, earth curvature is ignored for
    the interpoation. No extrapolation is applied for times that are out of
    bounds for the trajectories.
    """

    def __init__(self, trajectories: list[Trajectory]):
        self._trajectories = copy.deepcopy(trajectories)
        self._interpolants = []
        for trajectory in trajectories:
            x = [t.timestamp() for t in trajectory.times]
            y = np.stack(
                [
                    trajectory.lats,
                    trajectory.lons,
                    trajectory.alts,
                    trajectory.vlats,
                    trajectory.vlons,
                    trajectory.vzs,
                    trajectory.cross_sections,
                ],
                axis=1,
            )
            self._interpolants.append(CubicSpline(x, y, extrapolate=False))

    def get_targets(self, time: datetime.datetime) -> Iterable[Target]:
        targets: list[Target] = []
        for trajectory, f in zip(self._trajectories, self._interpolants):
            y = f(time.timestamp())
            if not np.isnan(y).any():
                assert len(y) == 7
                lat, lon, alt, vlat, vlon, vz, crs = y
                targets.append(
                    Target(
                        id=trajectory.target_id,
                        point=Point(
                            lat=lat,
                            lon=lon,
                            alt=alt,
                        ),
                        cross_section=crs,
                        vlat=vlat,
                        vlon=vlon,
                        vz=vz,
                    )
                )
        return targets
    
    def get_minimum_time(self) -> datetime.datetime:
        return np.min([t.times[0] for t in self._trajectories])
    
    def get_maximum_time(self) -> datetime.datetime:
        return np.max([t.times[-1] for t in self._trajectories])

    # def get_targets(self, time: datetime.datetime) -> Iterable[Target]:
    #     targets: list[Target] = []
    #     for trajectory in self._trajectories:
    #         if time < trajectory.times[0] or time > trajectory.times[-1]:
    #             continue
    #         i = next((i for i, t in enumerate(trajectory.times) if t > time), None)
    #         if i is None:
    #             raise RuntimeError("Unexpected case. This should never happen!")
    #         elif i == 0:
    #             lat = trajectory.lats[i]
    #             lon = trajectory.lons[i]
    #             alt = trajectory.alts[i]
    #             crs = trajectory.cross_sections[i]
    #             vlat = trajectory.vlats[i]
    #             vlon = trajectory.vlons[i]
    #             vz = trajectory.vzs[i]
    #         else:
    #             # Interpolate linearly.
    #             t = (time - trajectory.times[i - 1]) / (
    #                 trajectory.times[i] - trajectory.times[i - 1]
    #             )

    #             lat = trajectory.lats[i - 1] * (1 - t) + trajectory.lats[i] * t
    #             lon = trajectory.lons[i - 1] * (1 - t) + trajectory.lons[i] * t
    #             alt = trajectory.alts[i - 1] * (1 - t) + trajectory.alts[i] * t
    #             # fmt: off
    #             crs = trajectory.cross_sections[i - 1] * (1 - t) + trajectory.cross_sections[i] * t
    #             # fmt: on
    #             vlat = trajectory.vlats[i - 1] * (1 - t) + trajectory.vlats[i] * t
    #             vlon = trajectory.vlons[i - 1] * (1 - t) + trajectory.vlons[i] * t
    #             vz = trajectory.vzs[i - 1] * (1 - t) + trajectory.vzs[i] * t

    #         targets.append(
    #             Target(
    #                 id=trajectory.target_id,
    #                 point=Point(
    #                     lat=lat,
    #                     lon=lon,
    #                     alt=alt,
    #                 ),
    #                 cross_section=crs,
    #                 vlat=vlat,
    #                 vlon=vlon,
    #                 vz=vz,
    #             )
    #         )
    #     return targets
