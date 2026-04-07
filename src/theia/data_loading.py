import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from theia.coordinates import CoordinateTransformations
from theia.terrain import elevationAt
from theia.types import (
    AttenuationModel,
    Point,
    Polarization,
    Radar,
    Trajectory,
    Transmitter,
    Velocity,
    ConstantRcsModel,
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

        velocities: list[Velocity] = []
        for lat, lon, alt, vlat, vlon, vz in zip(
            rows["lat"],
            rows["lon"],
            rows["alt"],
            rows["vlat"],
            rows["vlon"],
            rows["vz"],
            strict=True,
        ):
            velocities.append(
                CoordinateTransformations.velocity_geodetic_to_cartesian(
                    p=Point(lat=lat, lon=lon, alt=alt),
                    vlat=vlat,
                    vlon=vlon,
                    valt=vz,
                )
            )

        assert len(rows["cross_section"].unique()) == 1
        rcs = rows["cross_section"].tolist()[0]

        trajectories.append(
            Trajectory(
                target_id=ID,
                times=rows["time"].to_numpy().astype("datetime64[ms]").tolist(),
                lats=rows["lat"].astype(float).tolist(),
                lons=rows["lon"].astype(float).tolist(),
                alts=rows["alt"].astype(float).tolist(),
                vxs=[v.vx for v in velocities],
                vys=[v.vy for v in velocities],
                vzs=[v.vz for v in velocities],
                cross_section_model=ConstantRcsModel(rcs=rcs),
            )
        )
        callsign_map[ID] = callsign
        ID += 1

    return trajectories, callsign_map


def _parse_attenuation_str(attenuation: str | float) -> list[float] | None:
    if type(attenuation) is not str and np.isnan(attenuation):
        return None
    if "VECTOR" not in attenuation:
        raise RuntimeError("Parsing failed")
    attenuation_values = [
        float(v) for v in attenuation.replace("VECTOR 1", "").strip().split(" ")
    ]  # dB

    return attenuation_values


def _build_transmitter(
    lat: float,
    lon: float,
    alt: float,
    antenna_height: float,
    frequency: float,
    bandwidth: float,
    erp_h: float,
    erp_v: float,
    polarization: Polarization,
    attenuation_values_h: list[float] | None,
    attenuation_values_v: list[float] | None,
) -> Transmitter:
    if attenuation_values_h is None:
        attenuation_model_horizontal = None
    else:
        angles = np.linspace(
            0,
            2 * np.pi,
            num=len(attenuation_values_h),
            endpoint=True,
        )
        assert len(attenuation_values_h) == 360
        attenuation_model_horizontal = AttenuationModel(
            attenuation_table_angles=angles,
            attenuation_table_values=attenuation_values_h,
            polarization=Polarization.HORIZONTAL,
        )

    if attenuation_values_v is None:
        attenuation_model_vertical = None
    else:
        angles = np.linspace(
            -np.pi / 2.0,
            np.pi / 2.0,
            num=len(attenuation_values_v),
            endpoint=True,
        )
        assert len(attenuation_values_v) == 181
        attenuation_model_vertical = AttenuationModel(
            attenuation_table_angles=angles,
            attenuation_table_values=attenuation_values_v,
            polarization=Polarization.VERTICAL,
        )

    return Transmitter(
        id=-1,
        point=Point(
            lat=lat,
            lon=lon,
            alt=alt,
        ),
        power=np.nan,
        erp=erp_h if not np.isnan(erp_h) else erp_v,
        antenna_height=antenna_height,
        antenna_diameter=np.nan,
        frequency=frequency,
        pulse_width=np.nan,
        polarization=polarization,
        bandwidth=bandwidth / 1000.0,
        vertical_attenuation=attenuation_model_vertical,
        horizontal_attenuation=attenuation_model_horizontal,
    )


def load_bakom_ukw_transmitters(
    path: str = f"{Path(__file__).parent.parent.parent / 'data' / 'BCSDR_CONCESSION_221208_CSV.csv'}",
    start_id: int = 0,
) -> list[Transmitter]:
    df = pd.read_csv(path, delimiter="\t")

    logging.warning("Not using altitude of BAKOM to ensure consistency within Theia!")

    failed = []
    transmitters = []
    for _, row in df.iterrows():
        try:
            name = f"{row['site_name']} {row['call_sign']} ({row['program.name']})"
            lat = row["Latitude"]
            lon = row["Longitude"]
            alt = row["site_alt"]
            antenna_height = row["hgt_agl"]
            frequency = row["frq_assign"]  # MHz
            bandwidth = row["bdwdth"]  # kHz
            erp_h = row["erp_h_w"]  # W
            erp_v = row["erp_v_w"]  # W
            polarization_str = row["polar"]

            assert np.logical_xor(np.isnan(erp_h), np.isnan(erp_v))

            if polarization_str == "H":
                polarization = Polarization.HORIZONTAL
            elif polarization_str == "V":
                polarization = Polarization.VERTICAL
            else:
                raise RuntimeError(f"Unknown polarization {polarization_str}")

            attenuation_values_h = _parse_attenuation_str(row["attn_h_h"])
            attenuation_values_v = _parse_attenuation_str(row["attn_h_v"])

            tx = _build_transmitter(
                lat,
                lon,
                elevationAt(lat, lon),
                antenna_height,
                frequency,
                bandwidth,
                erp_h,
                erp_v,
                polarization,
                attenuation_values_h,
                attenuation_values_v,
            )
            tx.id = start_id
            start_id += 1
            transmitters.append(tx)
        except:
            failed.append(name)

    if len(failed) > 0:
        logging.warning(
            f"Ignoring {len(failed)} / {df.shape[0]} ({len(failed) / df.shape[0] * 100:.1f}%) UKW transmitters because they have polarity other than 'V' or 'H'"
        )

    return transmitters


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
                cross_section_model=ConstantRcsModel(rcs=rcs),
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
    if start_time is None:
        start_time = min([t.times[0] for t in trajectories])
    if stop_time is None:
        stop_time = max([t.times[-1] for t in trajectories])

    dicts = []
    time = start_time
    while time <= stop_time:
        targets = [t(time) for t in trajectories]
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
