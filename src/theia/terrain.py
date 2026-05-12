import abc
import functools
import math
import os

import numba
import numpy as np
import pydantic

from theia.config import ELEVATION_DATA_DIR
from theia.distance import R_EARTH, haversine
from theia.types import Point


@functools.cache
def load_hgt_file(path: str):
    size = os.path.getsize(path)
    dim = int(math.sqrt(size / 2))
    assert dim**2 * 2 == size
    return np.fromfile(path, ">i2").reshape(dim, dim).astype(np.int16)


@numba.jit
def interpolate_elevation_tile(lat_f: float, lon_f: float, arr: np.ndarray) -> float:
    dim = arr.shape[0]
    # map to array coordinates
    i = (1 - lat_f) * (dim - 1)
    j = lon_f * (dim - 1)

    i0 = int(i)
    j0 = int(j)
    di = i - i0
    dj = j - j0

    # clamp edges
    i1 = min(i0 + 1, dim - 1)
    j1 = min(j0 + 1, dim - 1)

    # bilinear interpolation
    return (
        arr[i0, j0] * (1 - di) * (1 - dj)
        + arr[i1, j0] * di * (1 - dj)
        + arr[i0, j1] * (1 - di) * dj
        + arr[i1, j1] * di * dj
    )


# Inherits from pydantic.BaseModel so that other pdantic models can use a terrain model.
class AbstractTerrainModel(abc.ABC, pydantic.BaseModel):
    @abc.abstractmethod
    def elevationAt(lat: float, lon: float) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def has_line_of_sight(p1: Point, p2: Point) -> float:
        raise NotImplementedError()

    def sample_location(
        self,
        rng: np.random.Generator,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Point:
        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)
        alt = self.elevationAt(lat, lon)
        return Point(lat=lat, lon=lon, alt=alt)


class SrtmTerrainModel(AbstractTerrainModel):
    step_m: float = 30.0

    def elevationAt(self, lat: float, lon: float) -> float:
        lat0 = math.floor(lat)
        lon0 = math.floor(lon)

        ns = "N" if lat0 >= 0 else "S"
        ew = "E" if lon0 >= 0 else "W"
        filename = f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}.hgt"

        arr = load_hgt_file(f"{ELEVATION_DATA_DIR}/{filename}")

        # local fractional degree within tile
        lat_f = lat - lat0
        lon_f = lon - lon0

        return interpolate_elevation_tile(lat_f, lon_f, arr)

    def has_line_of_sight(self, p1: Point, p2: Point) -> float:
        """
        Checks line of sight between two points accounting for Earth curvature
        and terrain elevation.

        step_m controls sampling resolution along the path.
        """

        def rad(p):
            return math.radians(p.lat), math.radians(p.lon)

        lat1, lon1 = rad(p1)
        lat2, lon2 = rad(p2)

        distance = haversine(p1.lon, p1.lat, p2.lon, p2.lat)

        if distance == 0:
            return True

        steps = max(1, int(distance / self.step_m) + 1)

        # Heights above Earth's center
        h1 = R_EARTH + p1.alt
        h2 = R_EARTH + p2.alt

        for i in range(1, steps):
            t = i / steps

            # Interpolate along great circle
            A = math.sin((1 - t) * distance / R_EARTH) / math.sin(distance / R_EARTH)
            B = math.sin(t * distance / R_EARTH) / math.sin(distance / R_EARTH)

            x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(
                lon2
            )
            y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(
                lon2
            )
            z = A * math.sin(lat1) + B * math.sin(lat2)

            lat = math.atan2(z, math.sqrt(x * x + y * y))
            lon = math.atan2(y, x)

            lat_deg = math.degrees(lat)
            lon_deg = math.degrees(lon)

            # Terrain height above Earth's center
            terrain = R_EARTH + self.elevationAt(lat_deg, lon_deg)

            # Height of LOS ray at this point
            ray_height = (1 - t) * h1 + t * h2

            if terrain > ray_height:
                return False

        return True
