import datetime
import os
import numpy as np
import pandas as pd

from theia.types import (
    AttenuationModel,
    PassiveRadarDetection,
    Point,
    Polarization,
    Radar,
    Target,
)
from theia.util import erp_to_power


def load_pcl_reference_data(
    data_directory: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../tests/test_data/pcl_detection",
    rcs: float = 1,
) -> list[PassiveRadarDetection]:
    # Load detections.
    df_detections = pd.read_csv(f"{data_directory}/detections.csv")

    df_target = df_detections.loc[
        :, ["targ_id", "tgt_lat", "tgt_lon", "tgt_height", "vx", "vy", "vz"]
    ]
    df_target.drop_duplicates(inplace=True)

    targets = []
    for _, row in df_target.iterrows():
        targets.append(
            Target(
                id=row["targ_id"],
                point=Point(
                    lat=row["tgt_lat"],
                    lon=row["tgt_lon"],
                    alt=row["tgt_height"],
                ),
                cross_section=rcs,
                vlon=row["vx"],
                vlat=row["vy"],
                vz=row["vz"],
            )
        )

    # Load receivers.
    empty_attenuation_model = AttenuationModel(
        attenuation_table_angles=[],
        attenuation_table_values=[],
    )

    df_rx = pd.read_csv(f"{data_directory}/rx.csv")
    receivers = []
    for _, row in df_rx.iterrows():
        receivers.append(
            Radar(
                id=row["rx_id"],
                point=Point(
                    lat=row["lat"],
                    lon=row["lon"],
                    alt=row["masl"],
                ),
                power=0,
                erp=np.nan,
                antenna_height=row["ahmagl"],
                diameter=np.nan,
                frequency=np.nan,
                pulse_width=np.nan,
                cpi_pulses=0,
                bandwidth=row["bandwidth"] / 1000.0,  # convert kHz -> MHz
                pfa=np.nan,
                min_elevation=np.nan,
                max_elevation=np.nan,
                rotation_time=np.nan,
                polarization=Polarization.HORIZONTAL,  # dummy value
                gain=row["gain"],
                losses=row["losses"],
                noise_temperature=row["temp_sys"],
                vertical_attenuation=empty_attenuation_model,
                horizontal_attenuation=empty_attenuation_model,
            )
        )

    # Load transmitters.
    df_tx = pd.read_csv(f"{data_directory}/tx.csv")
    transmitters = []
    for _, row in df_tx.iterrows():
        is_horizontal = row["pol"] == "H"
        assert row["pol"] in ["H", "V"]

        erp = row["erp_h"] if is_horizontal else row["erp_v"]

        values = row["horiz_diagr_att"] if is_horizontal else row["vert_diagr_att"]
        if values == "UNDEFINED":
            values = []
        else:
            values = values.replace("{", "").replace("}", "")
            values = [float(v) for v in values.split(",")]

        angles = (
            np.linspace(0, 2 * np.pi, len(values))
            if is_horizontal
            else np.linspace(-np.pi / 2, np.pi / 2, len(values))
        )

        attenuation_model = AttenuationModel(
            attenuation_table_angles=angles,
            attenuation_table_values=values,
        )

        # Default values...
        losses = 0.
        gain = 0.
        antenna_diameter = 2.
        pulse_width = 1.
        cpi_pulses = 1
        pfa = 1e-6
        rotation_time = 1

        transmitters.append(
            Radar(
                id=row["tx_id"],
                point=Point(
                    lat=row["lat"],
                    lon=row["lon"],
                    alt=row["masl"],
                ),
                power=erp_to_power(erp, losses, gain),
                erp=erp,
                antenna_height=row["ahmagl"],
                diameter=antenna_diameter,
                frequency=row["freq"],
                pulse_width=pulse_width,
                cpi_pulses=cpi_pulses,
                bandwidth=row["bandwidth"] / 1000.0,  # convert kHz -> MHz
                pfa=pfa,
                min_elevation=np.nan,
                max_elevation=np.nan,
                rotation_time=rotation_time,
                polarization=Polarization.HORIZONTAL,  # dummy value
                vertical_attenuation=empty_attenuation_model
                if is_horizontal
                else attenuation_model,
                horizontal_attenuation=attenuation_model
                if is_horizontal
                else empty_attenuation_model,
            )
        )

    # Detections.
    df_detections = df_detections.loc[
        :, ["rx_id", "tx_id", "targ_id", "recording_time", "doppler", "range"]
    ]

    detections = []
    id = 0
    for _, row in df_detections.iterrows():
        detections.append(
            PassiveRadarDetection(
                detection_id=id,
                time=datetime.datetime.fromtimestamp(row["recording_time"]),
                transmitter=next(t for t in transmitters if t.id == row["tx_id"]),
                receiver=next(r for r in receivers if r.id == row["rx_id"]),
                target=next(t for t in targets if t.id == row["targ_id"]),
                bistatic_range=row["range"],
                doppler_shift=row["doppler"],
            )
        )
        id += 1

    return detections
