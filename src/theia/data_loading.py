import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd
from theia.coordinates import CoordinateTransformations
from theia.target_simulation.recorded_targets_simulator import RecordedTargetsSimulator
from theia.types import (
    AttenuationModel,
    Point,
    Polarization,
    Radar,
    Trajectory,
    Velocity,
)


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
        if rows.shape[0] < 2:
            logging.warning(
                f"Skip callsign {callsign} because only one time stamp is available"
            )
            continue

        for lat, lon, alt, vlat, vlon, vz in zip(
            rows["lat"],
            rows["lon"],
            rows["alt"],
            rows["vlat"],
            rows["vlon"],
            rows["vz"],
            strict=True,
        ):
            CoordinateTransformations.velocity_geodetic_to_cartesian(
                p=Point(lat=lat, lon=lon, alt=alt),
                vlat=vlat,
                vlon=vlon,
                valt=vz,
            )

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


def _load_transmitter_of_opportunity(series):
    assert series["polar"] in ["V", "H"]

    erp = series["erp_v_w"]
    if series["polar"] == "H":
        erp = series["erp_h_w"]
    polarization = Polarization.VERTICAL
    if series["polar"] == "H":
        polarization = Polarization.HORIZONTAL

    horizontal_attenuation_model = None
    if type(series["attn_h_h"]) is str:
        values = [float(v) for v in series["attn_h_h"].replace("VECTOR", "").split()]
        horizontal_angles = np.linspace(0, 2 * np.pi, len(values))
        horizontal_attenuation_model = AttenuationModel(
            attenuation_table_angles=horizontal_angles,
            attenuation_table_values=values,
        )

    vertical_attenuation_model = None
    if type(series["attn_h_v"]) is str:
        values = [float(v) for v in series["attn_h_v"].replace("VECTOR", "").split()]
        vertical_angles = np.linspace(-np.pi / 2, np.pi / 2, len(values))
        vertical_attenuation_model = AttenuationModel(
            attenuation_table_angles=vertical_angles,
            attenuation_table_values=values,
        )

    return Radar(
        id=-1,
        point=Point(
            lat=series["Latitude"],
            lon=series["Longitude"],
            alt=series["site_alt"],
        ),
        power=erp,
        frequency=series["frq_assign"],
        erp=erp,  # Is this correct or do I need some unit conversion???
        antenna_height=series["hgt_agl"],
        diameter=2.0,  # not really needed
        pulse_width=1.0,  # not really needed
        cpi_pulses=1,  # not really needed
        bandwidth=series["bdwdth"],
        pfa=1e-6,  # not really needed?
        min_elevation=-20.0,  # not really needed
        max_elevation=60.0,  # not really needed
        rotation_time=10.0,  # not really needed
        polarization=polarization,
        horizontal_attenuation=horizontal_attenuation_model,
        vertical_attenuation_model=vertical_attenuation_model,
    )


def load_transmitters_of_opportunity(file: str) -> dict[str, Radar]:
    df = pd.read_csv(file, delimiter=";").set_index("site_name")
    return {
        name: _load_transmitter_of_opportunity(series) for name, series in df.iterrows()
    }


def load_openburst_trajectory_file(path: str, rcs: float = 1.0) -> list[Trajectory]:
    cols = [
        "DateTimeIndex",
        "millisecs",
        "converted_integer_id",
        "lat",
        "lon",
        "heading[0 = north\n180 = south\n360 = north]",
        "speed [km / h]",
        "altitude[m]",
        "track_quality",
        "milli_secs_after_midnight",
        "tgt_vx [vel m/s on lon axis]",
        "tgt_vy [vel m/s on lat axis]",
        "tgt_vz [vel m/s on z axis]",
    ]

    df = pd.DataFrame(np.load(path, allow_pickle=True), columns=cols)

    assert len(df.loc[:, "DateTimeIndex"].unique() == 1)
    t0 = df.loc[:, "DateTimeIndex"].unique()[0]

    times = [
        (t0 + datetime.timedelta(microseconds=ms * 1000)).to_pydatetime()
        for ms in df.loc[:, "milli_secs_after_midnight"]
    ]

    trajectories = []
    for ID, df_target in df.groupby("converted_integer_id"):
        df_target.sort_values("milli_secs_after_midnight", inplace=True)
        velocities: list[Velocity] = []
        for _, row in df_target.iterrows():
            velocities.append(
                CoordinateTransformations.velocity_geodetic_to_cartesian(
                    p=Point(
                        lat=row["lat"],
                        lon=row["lon"],
                        alt=row["altitude[m]"],
                    ),
                    # TODO not sure whether this is actually correct...
                    # See #20 and #23 in openBURST
                    vlat=row["tgt_vx [vel m/s on lon axis]"],
                    vlon=row["tgt_vy [vel m/s on lat axis]"],
                    valt=row["tgt_vz [vel m/s on z axis]"],
                )
            )
        trajectories.append(
            Trajectory(
                target_id=ID,
                times=times,
                lats=df_target.loc[:, "lat"],
                lons=df_target.loc[:, "lon"],
                alts=df_target.loc[:, "altitude[m]"],
                vxs=[v.vx for v in velocities],
                vys=[v.vy for v in velocities],
                vzs=[v.vz for v in velocities],
                cross_sections=[rcs for _ in range(df_target.shape[0])],
            )
        )

    return trajectories


def save_openburst_trajectory_file(
    output_path: str,
    trajectories: list[Trajectory],
    start_time: Optional[datetime.datetime] = None,
    stop_time: Optional[datetime.datetime] = None,
    dt: datetime.timedelta = datetime.timedelta(seconds=10),
):
    sim = RecordedTargetsSimulator(trajectories)
    if start_time is None:
        start_time = sim.get_minimum_time()
    if stop_time is None:
        stop_time = sim.get_maximum_time()

    dicts = []
    time = start_time
    while time <= stop_time:
        targets = sim.get_targets(time)
        for target in targets:
            vlat, vlon, vz = CoordinateTransformations.velocity_cartesian_to_geodetic(
                target.point,
                target.velocity,
            )
            dicts.append(
                {
                    "DateTimeIndex": time,
                    "millisecs": 0,
                    "milli_secs_after_midnight": (time - start_time).total_seconds()
                    * 1000,
                    "converted_integer_id": target.id,
                    "lat": target.lat,
                    "lon": target.lon,
                    "altitude[m]": target.alt,
                    "track_quality": 1,
                    "heading[0 = north\n180 = south\n360 = north]": 0,
                    "speed [km / h]": target.velocity.speed * 1000 / 3600,
                    "tgt_vx [vel m/s on lon axis]": vlon,
                    "tgt_vy [vel m/s on lat axis]": vlat,
                    "tgt_vz [vel m/s on z axis]": vz,
                }
            )
        time = time + dt
    df = pd.DataFrame(dicts)

    cols = [
        "DateTimeIndex",
        "millisecs",
        "converted_integer_id",
        "lat",
        "lon",
        "heading[0 = north\n180 = south\n360 = north]",
        "speed [km / h]",
        "altitude[m]",
        "track_quality",
        "milli_secs_after_midnight",
        "tgt_vx [vel m/s on lon axis]",
        "tgt_vy [vel m/s on lat axis]",
        "tgt_vz [vel m/s on z axis]",
    ]

    df = df.loc[:, cols]

    np.save(output_path, df.values)
