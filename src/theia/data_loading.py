import os
import math
import functools

import numpy as np
import numba
from theia.config import ELEVATION_DATA_DIR


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


def elevationAt(lat: float, lon: float) -> float:
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
