import datetime
import logging
import os
import numpy as np
import pandas as pd

from theia.coordinates import (
    CoordinateTransformations,
)
from theia.data_loading import load_openburst_trajectory_file
from theia.types import (
    AttenuationModel,
    PassiveRadarDetection,
    Point,
    Polarization,
    Radar,
    Receiver,
    Target,
    Trajectory,
    Transmitter,
)
from theia.util import erp_to_power, from_dB


def load_pcl_reference_data(
    data_directory: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../tests/test_data/pcl_detection",
    rcs: float = 1,
) -> tuple[list[Trajectory], list[tuple[PassiveRadarDetection, float]]]:
    # Load trajectories.
    trajectories = load_openburst_trajectory_file(f"{data_directory}/replay.npy")
    t_min = min([t.times[0] for t in trajectories])

    # Load detections.
    df_detections = pd.read_csv(f"{data_directory}/detections.csv")

    # Load receivers.
    empty_model_horizontal = AttenuationModel(
        attenuation_table_angles=[],
        attenuation_table_values=[],
        polarization=Polarization.HORIZONTAL,
    )
    empty_model_vertical = AttenuationModel(
        attenuation_table_angles=[],
        attenuation_table_values=[],
        polarization=Polarization.VERTICAL,
    )

    df_rx = pd.read_csv(f"{data_directory}/rx.csv")
    receivers = []
    for _, row in df_rx.iterrows():
        receivers.append(
            Receiver(
                id=row["rx_id"],
                point=Point(
                    lat=row["lat"],
                    lon=row["lon"],
                    alt=row["masl"],
                ),
                antenna_height=row["ahmagl"],
                diameter=np.nan,
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
                vertical_attenuation=empty_model_vertical,
                horizontal_attenuation=empty_model_horizontal,
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

        if is_horizontal:
            horizontal_model = AttenuationModel(
                attenuation_table_angles=angles,
                attenuation_table_values=values,
                polarization=Polarization.HORIZONTAL,
            )
            vertical_model = empty_model_vertical
        else:
            horizontal_model = empty_model_horizontal
            vertical_model = AttenuationModel(
                attenuation_table_angles=angles,
                attenuation_table_values=values,
                polarization=Polarization.VERTICAL,
            )

        # Default values...
        losses = 0.0
        gain = 0.0
        antenna_diameter = 2.0
        pulse_width = 1.0
        cpi_pulses = 1
        pfa = 1e-6
        rotation_time = 1

        transmitters.append(
            Transmitter(
                id=row["tx_id"],
                point=Point(
                    lat=row["lat"],
                    lon=row["lon"],
                    alt=row["masl"],
                ),
                power=erp_to_power(erp, losses, gain),
                erp=from_dB(erp),
                antenna_height=row["ahmagl"],
                antenna_diameter=np.nan,
                frequency=row["freq"],
                pulse_width=pulse_width,
                polarization=Polarization.HORIZONTAL,  # dummy value
                bandwidth=row["bandwidth"] / 1000.0,  # convert kHz -> MHz
                vertical_attenuation=vertical_model,
                horizontal_attenuation=horizontal_model,
            )
        )

    # Detections.
    df_detections = df_detections.loc[
        :,
        [
            "rx_id",
            "tx_id",
            "targ_id",
            "recording_time",
            "doppler",
            "range",
            "snr",
            "tgt_lat",
            "tgt_lon",
            "tgt_height",
            "vx",
            "vy",
            "vz",
        ],
    ]
    df_detections.drop_duplicates(inplace=True)

    detections = []
    id = 0
    for _, row in df_detections.iterrows():
        if row["recording_time"] == 0:
            logging.info("Skipping first detection because its velocities are zero.")
            assert np.isclose(row["vx"], 0)
            assert np.isclose(row["vy"], 0)
            assert np.isclose(row["vz"], 0)
            continue
        target_position = Point(
            lat=row["tgt_lat"],
            lon=row["tgt_lon"],
            alt=row["tgt_height"],
        )
        # Careful: vx, vy, vz are NOT in Cartesian coordinates in openBURST!
        vlon = row["vx"]
        vlat = row["vy"]
        valt = row["vz"]

        velocity = CoordinateTransformations.velocity_geodetic_to_cartesian(
            target_position,
            vlat,
            vlon,
            valt,
        )

        detections.append(
            (
                PassiveRadarDetection(
                    detection_id=id,
                    time=t_min + datetime.timedelta(milliseconds=row["recording_time"]),
                    radar=Radar(
                        transmitter=next(
                            t for t in transmitters if t.id == row["tx_id"]
                        ),
                        receiver=next(r for r in receivers if r.id == row["rx_id"]),
                    ),
                    target=Target(
                        id=int(row["targ_id"]),
                        point=target_position,
                        cross_section=rcs,
                        velocity=velocity,
                    ),
                    bistatic_range=row["range"] * 1000,
                    doppler_shift=row["doppler"],
                ),
                row["snr"],
            )
        )
        id += 1

    return trajectories, detections
