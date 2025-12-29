import os
import math
import functools

import numpy as np
import numba
import pandas as pd
from theia.config import ELEVATION_DATA_DIR
from theia.types import Trajectory


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


def load_trajectory_file(path: str) -> tuple[list[Trajectory], dict[int, str]]:
    """
    Load trajectories from the CSV file at the given path.

    Returns
    -------
    trajectories: list[Trajectory]
        the trajectories
    target_callsigns: dict[int, str]
        lookup table that maps the target ID to the corresponding callsign
    """
    df = pd.read_csv(path)
    expected_columns = [
        "icao24",
        "time",
        "lat",
        "lon",
        "alt",
        "vlat",
        "vlon",
        "vz",
        "cross_section",
        "callsign",
        "manufacturerIcao",
        "model",
    ]
    assert (df.columns == expected_columns).all()
    df.sort_values(["callsign", "time"], inplace=True)
    df["time"] = pd.to_datetime(df["time"])

    trajectories: list[Trajectory] = []
    callsign_map: dict[int, str] = {}
    ID = 0
    for callsign, rows in df.groupby("callsign"):
        trajectories.append(
            Trajectory(
                target_id=ID,
                times=rows["time"].to_numpy().astype("datetime64[ms]").tolist(),
                lats=rows["lat"].astype(float).tolist(),
                lons=rows["lon"].astype(float).tolist(),
                alts=rows["alt"].astype(float).tolist(),
                vlats=rows["vlat"].astype(float).tolist(),
                vlons=rows["vlon"].astype(float).tolist(),
                vzs=rows["vz"].astype(float).tolist(),
                cross_sections=rows["cross_section"].astype(float).tolist(),
            )
        )
        callsign_map[ID] = callsign
        ID += 1

    return trajectories, callsign_map
