import abc
import datetime
import itertools
import json

import numpy as np
import pandas as pd
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from theia.coordinates import CoordinateTransformations
from theia.stonesoup_interface import MonostaticDetectionFactory
from theia.types import (
    ConstantRcsModel,
    MonostaticRadarDetection,
    Radar,
    Receiver,
    SituationalPicture,
    Snapshot,
    Target,
    Trajectory,
    Transmitter,
)


class AbstractSimulationListener(abc.ABC):
    @abc.abstractmethod
    def on_snapshot(self, snapshot: Snapshot):
        raise NotImplementedError()

    @abc.abstractmethod
    def on_situational_picture(
        self,
        situational_picture: SituationalPicture,
        is_blue: bool,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def on_detections(
        self,
        active_radar_detections: list[MonostaticRadarDetection],
        is_blue: bool,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def on_end(self):
        raise NotImplementedError()


class NoLogger(AbstractSimulationListener):
    def on_snapshot(self, snapshot):
        pass

    def on_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        pass

    def on_detections(
        self, active_radar_detections: list[MonostaticRadarDetection], is_blue: bool
    ):
        pass

    def on_end(self):
        pass


class FileLogger(AbstractSimulationListener):
    def __init__(self, path: str, override: bool = True):
        self._path = path
        self._override = override
        self.snapshots = []
        self.situational_pictures = []
        self.active_radar_detections = []

    def on_snapshot(self, snapshot):
        self.snapshots.append(snapshot.model_dump(mode="json"))

    def on_situational_picture(
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

    def on_detections(
        self,
        active_radar_detections: list[MonostaticRadarDetection],
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

    def on_end(self):
        with open(self._path, "a" if not self._override else "w") as file:
            json.dump(
                {
                    # "situational_pictures": self.situational_pictures,
                    "snapshots": self.snapshots,
                    "active_radar_detections": self.active_radar_detections,
                },
                file,
            )


class PrintLogger(AbstractSimulationListener):
    def on_snapshot(self, snapshot):
        print(snapshot)

    def on_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        print(f"Situational picture {'BLUE' if is_blue else 'RED'}:")
        print(situational_picture)

    def on_detections(
        self, active_radar_detections: list[MonostaticRadarDetection], is_blue: bool
    ):
        print(f"Active radar detections {'BLUE' if is_blue else 'RED'}:")
        print(active_radar_detections)

    def on_end(self):
        pass


class InMemoryLogger(AbstractSimulationListener):
    def __init__(self):
        self.situational_pictures_blue: list[SituationalPicture] = []
        self.situational_pictures_red: list[SituationalPicture] = []
        self.monostatic_radar_detections_blue: list[MonostaticRadarDetection] = []
        self.snapshots: list[Snapshot] = []

    def on_snapshot(self, snapshot):
        self.snapshots.append(snapshot)

    def on_situational_picture(
        self,
        situational_picture: SituationalPicture,
        is_blue: bool,
    ):
        pictures = (
            self.situational_pictures_blue if is_blue else self.situational_pictures_red
        )
        pictures.append(situational_picture)

    def on_detections(
        self,
        active_radar_detections: list[MonostaticRadarDetection],
        is_blue: bool,
    ):
        if not is_blue:
            raise NotImplementedError()
        for det in active_radar_detections:
            self.monostatic_radar_detections_blue.append(det)

    def on_end(self):
        pass


class LogLoader:
    """Load JSON file written by FileLogger."""

    def __init__(self, path: str):
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
            MonostaticRadarDetection.model_validate(d) for d in blue_detections
        ]
        self._raw_blue_detections = sorted(
            blue_detections, key=lambda d: (d.time, d.radar.receiver.id, d.target.id)
        )
        self._blue_monostatic_radar_detections: list[Detection] = []
        for detection in blue_detections:
            self._blue_monostatic_radar_detections.append(
                MonostaticDetectionFactory.from_theia(detection)
            )


class CompositeSimulationListener(AbstractSimulationListener):
    def __init__(self, listeners: list[AbstractSimulationListener]):
        self._listeners = listeners

    def on_snapshot(self, snapshot: Snapshot):
        for listener in self._listeners:
            listener.on_snapshot(snapshot)

    def on_situational_picture(
        self,
        situational_picture: SituationalPicture,
        is_blue: bool,
    ):
        for listener in self._listeners:
            listener.on_situational_picture(situational_picture, is_blue)

    def on_detections(
        self,
        active_radar_detections: list[MonostaticRadarDetection],
        is_blue: bool,
    ):
        for listener in self._listeners:
            listener.on_detections(active_radar_detections, is_blue)

    def on_end(self):
        for listener in self._listeners:
            listener.on_end()


class FilterSimulationListener(AbstractSimulationListener):
    def __init__(
        self,
        listener: AbstractSimulationListener,
        forward_snapshots: bool,
        forward_situational_pictures: bool,
        forward_detections: bool,
    ):
        self._listener = listener
        self._forward_snapshots = forward_snapshots
        self._forward_detections = forward_detections
        self._forward_situational_pictures = forward_situational_pictures

    def on_snapshot(self, snapshot):
        if self._forward_snapshots:
            self._listener.on_snapshot(snapshot)

    def on_situational_picture(self, situational_picture, is_blue):
        if self._forward_situational_pictures:
            self._listener.on_situational_picture(situational_picture, is_blue)

    def on_detections(self, active_radar_detections, is_blue):
        if self._forward_detections:
            self._listener.on_detections(active_radar_detections, is_blue)

    def on_end(self):
        self._listener.on_end()


class SituationalPictureBuffer(AbstractSimulationListener):
    def __init__(self):
        self._red_ground_truth_history: dict[
            int, list[tuple[datetime.datetime, Target]]
        ] = {}
        # TODO: Implement for blue targets as well.
        self._situational_picture_blue = SituationalPicture(
            time=datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc),
            friendly_radars=[],
            friendly_targets=[],
            enemy_targets=[],
        )
        self._situational_picture_red = SituationalPicture(
            time=datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc),
            friendly_radars=[],
            friendly_targets=[],
            enemy_targets=[],
        )
        self._has_completed = False

    def on_snapshot(self, snapshot: Snapshot):
        for target in snapshot.red_targets:
            traj = self._red_ground_truth_history.get(target.id, [])
            traj.append((snapshot.time, target))
            self._red_ground_truth_history[target.id] = traj
        # TODO: Implement for blue targets.

    def on_situational_picture(
        self,
        situational_picture: SituationalPicture,
        is_blue: bool,
    ):
        if is_blue:
            self._situational_picture_blue = situational_picture
        else:
            self._situational_picture_red = situational_picture

    def on_detections(
        self,
        active_radar_detections: list[MonostaticRadarDetection],
        is_blue: bool,
    ):
        pass

    def on_end(self):
        self._has_completed = True

    def get_ground_truth_trajectories(self, is_blue: bool) -> list[Trajectory]:
        if is_blue:
            raise NotImplementedError()
        else:
            history = self._red_ground_truth_history.values()
        trajectories = []
        for target_history in history:
            times = []
            lats = []
            lons = []
            alts = []
            vxs = []
            vys = []
            vzs = []
            if len(target_history) < 2:
                continue
            for time, target in target_history:
                times.append(time)
                lats.append(target.lat)
                lons.append(target.lon)
                alts.append(target.alt)
                vxs.append(target.velocity.vx)
                vys.append(target.velocity.vy)
                vzs.append(target.velocity.vz)
            trajectories.append(
                Trajectory(
                    target_id=target.id,
                    times=times,
                    lats=lats,
                    lons=lons,
                    alts=alts,
                    vxs=vxs,
                    vys=vys,
                    vzs=vzs,
                    cross_section_model=ConstantRcsModel(rcs=np.nan),
                )
            )
        return trajectories

    def get_situational_picture(self, is_blue: bool) -> SituationalPicture:
        if is_blue:
            return self._situational_picture_blue
        else:
            return self._situational_picture_red

    def has_comleted(self) -> bool:
        return self._has_completed
