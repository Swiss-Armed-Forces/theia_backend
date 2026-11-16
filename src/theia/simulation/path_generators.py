"""
Strategies that turn a launch (start point, destination, launch time, speed)
into a fixed flight path.

Each generator is a pydantic model, so a player can choose one (and its
parameters) as part of a :class:`theia.simulation.actions.LaunchAction`.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.distance import line_of_sight_distance
from theia.types import Point
def build_trajectory(alpha: float, p_start: Point, p_stop: Point, dt: int = 1):
    """
    Calculate a ballistic trajectory that connects the start and stop point.

    The only force considered is gravity.

    Parameters
    ----------
    alpha: float
        Start angle of the ballistic motion [rad] in the 2D coordinate system
        (see "Returns")
    p_start: Point
        Start point of the trajectory
    p_stop: Point
        Stop point of the trajectory
    dt: int
        Number of seconds to use for representing the trajectory

    Returns
    -------
    v0: float
        Initial velocity [m/s]
    times: np.ndarray
        Time steps of the trajectory [s]. Shape (N,).
    trajectory: np.ndarray
        Points along the trajectory at 10s resolution. Shape (N, 2).
        Column 0 contains the horizontal component, column 1 the vertical component
        of the trajectory.
        The coordinate system is spanned by the LOS from start to end point and
        the parto of radial direction at the start point that is orthogonal to the LOS.
    """
    distance = line_of_sight_distance(*p_start.as_tuple(), *p_stop.as_tuple())

    p_start_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_start.as_tuple())
    )
    p_stop_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_stop.as_tuple())
    )
    direction = p_stop_ecef - p_start_ecef
    direction /= np.linalg.norm(direction)

    v0 = np.sqrt(9.81 * distance / (2 * np.cos(alpha) * np.sin(alpha)))
    t_stop = 2 * np.sin(alpha) * v0 / 9.81

    times = np.arange(0, t_stop, dt).tolist() + [t_stop]
    xs = []
    ys = []
    for t in times:
        x = np.cos(alpha) * v0 * t
        y = np.sin(alpha) * v0 * t - 0.5 * 9.81 * t**2
        xs.append(x)
        ys.append(y)

    if t_stop % 1 != 0:
        times.append(np.ceil(t_stop))
        xs.append(xs[-1])
        ys.append(ys[-1])

    return v0, times, np.vstack([xs, ys]).T


def convert_to_geodetic(
    p_start: Point,
    p_stop: Point,
    trajectory: np.ndarray,
) -> list[Point]:
    p_start_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_start.as_tuple())
    )
    p_stop_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_stop.as_tuple())
    )

    x_direction = p_stop_ecef - p_start_ecef
    x_direction /= np.linalg.norm(x_direction)
    y_direction = p_start_ecef / np.linalg.norm(p_start_ecef)
    y_direction -= np.dot(x_direction, y_direction) * x_direction
    assert np.isclose(np.dot(x_direction, y_direction), 0)

    points: list[Point] = []
    for x, y in trajectory:
        p_ecef = p_start_ecef + x * x_direction + y * y_direction
        p = CoordinateTransformations.cartesian_to_geodetic(*p_ecef)
        points.append(Point(lat=p[0], lon=p[1], alt=p[2]))
    return points
