import abc
import itertools
import json

import pandas as pd
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from theia.coordinates import CoordinateTransformations
from theia.types import ActiveRadarDetection, Radar, Receiver, SituationalPicture, Snapshot, Transmitter


class AbstractSimulationLogger(abc.ABC):
    @abc.abstractmethod
    def log_snapshot(self, snapshot: Snapshot):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_situational_picture(
        self,
        situational_picture: SituationalPicture,
        is_blue: bool,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_detections(
        self,
        active_radar_detections: list[ActiveRadarDetection],
        is_blue: bool,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def end(self):
        raise NotImplementedError()


class NoLogger(AbstractSimulationLogger):
    def log_snapshot(self, snapshot):
        pass

    def log_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        pass

    def log_detections(
        self, active_radar_detections: list[ActiveRadarDetection], is_blue: bool
    ):
        pass

    def end(self):
        pass


class FileLogger(AbstractSimulationLogger):
    def __init__(self, path: str, override: bool = True):
        self._path = path
        self._override = override
        self.snapshots = []
        self.situational_pictures = []
        self.active_radar_detections = []

    def log_snapshot(self, snapshot):
        self.snapshots.append(snapshot.model_dump(mode="json"))

    def log_situational_picture(
        self,
        situational_picture: SituationalPicture,
        is_blue: bool,
    ):
        self.situational_pictures.append(
            {
                "team": "blue" if is_blue else "red",
                "situational_picture": situational_picture.model_dump(mode="json"),
            }
        )

    def log_detections(
        self,
        active_radar_detections: list[ActiveRadarDetection],
        is_blue: bool,
    ):
        self.active_radar_detections.append(
            {
                "team": "blue" if is_blue else "red",
                "active_radar_detections": [
                    d.model_dump(mode="json") for d in active_radar_detections
                ],
            }
        )

    def end(self):
        with open(self._path, "a" if not self._override else "w") as file:
            json.dump(
                {
                    # "situational_pictures": self.situational_pictures,
                    "snapshots": self.snapshots,
                    "active_radar_detections": self.active_radar_detections,
                },
                file,
            )


class PrintLogger(AbstractSimulationLogger):
    def log_snapshot(self, snapshot):
        print(snapshot)

    def log_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        print(f"Situational picture {'BLUE' if is_blue else 'RED'}:")
        print(situational_picture)

    def log_detections(
        self, active_radar_detections: list[ActiveRadarDetection], is_blue: bool
    ):
        print(f"Active radar detections {'BLUE' if is_blue else 'RED'}:")
        print(active_radar_detections)

    def end(self):
        pass


class InMemoryLogger(AbstractSimulationLogger):
    def __init__(self):
        self.situational_pictures = []
        self.active_radar_detections = []
        self.snapshots = []

    def log_snapshot(self, snapshot):
        self.snapshots.append(snapshot)

    def log_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        self.situational_pictures.append(
            {
                "is_blue": is_blue,
                "situational_picture": situational_picture.model_dump(mode="json"),
            }
        )

    def log_detections(
        self, active_radar_detections: list[ActiveRadarDetection], is_blue: bool
    ):
        for det in active_radar_detections:
            self.active_radar_detections.append(
                {
                    "is_blue": is_blue,
                    "detection": det.model_dump(mode="json"),
                }
            )

    def end(self):
        pass


class LogLoader:
    """Load JSON file written by FileLogger."""
    def __init__(self, path: str, measurement_model: MeasurementModel):
        self._measurement_model = measurement_model
        with open(path, "r") as file:
            self._data = json.load(file)
        self._load_snapshots()
        self._load_monostatic_radars(is_blue=True)
        self._load_target_ground_truth(is_blue=False)
        self._load_detections(is_blue=True)

    @property
    def blue_monostatic_radars(self) -> list[Radar]:
        return self._blue_monostatic_radars

    @property
    def blue_monostatic_radar_detections(self) -> list[Detection]:
        return self._blue_monostatic_radar_detections

    @property
    def red_target_ground_truth(self) -> dict[int, GroundTruthPath]:
        return self._red_target_ground_truth

    def _load_snapshots(self):
        self._snapshots = [Snapshot.model_validate(d) for d in self._data["snapshots"]]

    def _load_monostatic_radars(self, is_blue: bool):
        if not is_blue:
            raise NotImplementedError()

        blue_transmitters: dict[int, Transmitter] = {}
        blue_receivers: dict[int, Receiver] = {}
        for snapshot in self._snapshots:
            blue_transmitters.update({tx.id: tx for tx in snapshot.blue_transmitters})
            blue_receivers.update({rx.id: rx for rx in snapshot.blue_receivers})

        self._blue_monostatic_radars = []
        for tx, rx in itertools.product(
            blue_transmitters.values(), blue_receivers.values()
        ):
            if tx.point == rx.point:
                self._blue_monostatic_radars.append(Radar(transmitter=tx, receiver=rx))

    def _load_target_ground_truth(self, is_blue: bool):
        if is_blue:
            raise NotImplementedError()

        red_target_states = []
        for snapshot in self._snapshots:
            for target in snapshot.red_targets:
                x, y, z = CoordinateTransformations.geodetic_to_cartesian(
                    *target.point.as_tuple()
                )
                red_target_states.append(
                    {
                        "id": target.id,
                        "time": snapshot.time,
                        "x": x,
                        "y": y,
                        "z": z,
                        "vx": target.velocity.vx,
                        "vy": target.velocity.vy,
                        "vz": target.velocity.vz,
                    }
                )
        red_target_states = pd.DataFrame(red_target_states).sort_values(["id", "time"])

        self._red_target_ground_truth: dict[int, GroundTruthPath] = {}
        for target_id, target_rows in red_target_states.groupby("id"):
            ground_truth_path = GroundTruthPath()
            for _, row in target_rows.iterrows():
                ground_truth_path.append(
                    GroundTruthState(
                        row[["x", "vx", "y", "vy", "z", "vz"]].values,
                        timestamp=row["time"],
                        metadata={"target_id": target_id},
                    )
                )
            self._red_target_ground_truth[target_id] = ground_truth_path

    def _load_detections(self, is_blue: bool):
        if not is_blue:
            raise NotImplementedError()

        blue_detections = list(
            itertools.chain.from_iterable(
                [
                    det["active_radar_detections"]
                    for det in self._data["active_radar_detections"]
                    if det["team"] == "blue"
                ]
            )
        )
        blue_detections = [
            ActiveRadarDetection.model_validate(d) for d in blue_detections
        ]
        properties = []
        for detection in blue_detections:
            x, y, z = CoordinateTransformations.geodetic_to_cartesian(
                *detection.target.point.as_tuple()
            )
            properties.append(
                {
                    "id": detection.detection_id,
                    "radar_id": detection.radar.receiver.id,
                    "target_id": detection.target.id,
                    "time": detection.time,
                    "range": detection.target_range,
                    "elevation": detection.elevation_angle,
                    "azimuth": detection.azimuth_angle,
                    "target_x": x,
                    "target_y": y,
                    "target_z": z,
                    "target_vx": detection.target.velocity.vx,
                    "target_vy": detection.target.velocity.vy,
                    "target_vz": detection.target.velocity.vz,
                }
            )

        df_blue_detections = pd.DataFrame(properties).sort_values(
            ["time", "radar_id", "target_id"]
        )

        self._blue_monostatic_radar_detections: list[Detection] = []
        for _, row in df_blue_detections.iterrows():
            self._blue_monostatic_radar_detections.append(
                Detection(
                    state_vector=row[
                        [
                            "target_x",
                            "target_y",
                            "target_z",
                        ]
                    ],
                    timestamp=row["time"],
                    metadata={
                        "radar_id": row["radar_id"],
                        "target_id": row["target_id"],
                    },
                    measurement_model=self._measurement_model,
                )
            )