import datetime
import itertools
import json

import numpy as np
import pandas as pd
from pydantic import TypeAdapter
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from theia.coordinates import CoordinateTransformations
from theia.simulation.simulator import AbstractSimulationListener, Simulator
from theia.types import (
    ConstantRcsModel,
    Event,
    GeoJSONFeature,
    KillEvent,
    MonostaticRadarDetection,
    MonostaticSensor,
    PclDetection,
    PclSensor,
    PetDetection,
    Shot,
    SituationalPicture,
    Snapshot,
    Target,
    TextEvent,
    TrackInitEvent,
    Trajectory,
)


class NoLogger(AbstractSimulationListener):
    def register_simulator(self, simulator: Simulator):
        pass

    def on_snapshot(self, snapshot):
        pass

    def on_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        pass

    def on_detections(
        self,
        active_radar_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        is_blue: bool,
    ):
        pass

    def on_end(self):
        pass

    def on_events(self, events: list[Event]):
        pass


class FileLogger(AbstractSimulationListener):
    def __init__(
        self,
        path: str,
        override: bool = True,
        log_situational_picture: bool = False,
    ):
        self._path = path
        self._override = override
        self._log_situational_picture = log_situational_picture
        self.snapshots = []
        self.situational_pictures = []
        self.detections = []
        self.events: list[Event] = []

    def register_simulator(self, simulator: Simulator):
        pass

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
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        is_blue: bool,
    ):
        self.detections.append(
            {
                "team": "blue" if is_blue else "red",
                "active_radar_detections": [
                    d.model_dump(mode="json") for d in active_radar_detections
                ],
                "pcl_detections": [d.model_dump(mode="json") for d in pcl_detections],
                "pet_detections": [d.model_dump(mode="json") for d in pet_detections],
            }
        )

    def on_events(self, events: list[Event]):
        self.events.extend(events)

    def on_end(self):
        data = {
            "snapshots": self.snapshots,
            "detections": self.detections,
            "events": [e.model_dump(mode="json") for e in self.events],
        }
        if self._log_situational_picture:
            data["situational_pictures"] = self.situational_pictures
        text = json.dumps(data)
        with open(self._path, "a" if not self._override else "w") as file:
            file.write(text)


class PrintLogger(AbstractSimulationListener):
    def register_simulator(self, simulator: Simulator):
        print("Register simulator")

    def on_snapshot(self, snapshot):
        print(snapshot)

    def on_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        print(f"Situational picture {'BLUE' if is_blue else 'RED'}:")
        print(situational_picture)

    def on_detections(
        self,
        active_radar_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        is_blue: bool,
    ):
        print(f"Active radar detections {'BLUE' if is_blue else 'RED'}:")
        print(active_radar_detections)
        print(f"PCL detections {'BLUE' if is_blue else 'RED'}:")
        print(pcl_detections)
        print(f"PET detections {'BLUE' if is_blue else 'RED'}:")
        print(pet_detections)

    def on_events(self, events: list[Event]):
        print(events)

    def on_end(self):
        pass


class InMemoryLogger(AbstractSimulationListener):
    def __init__(self):
        self.situational_pictures_blue: list[SituationalPicture] = []
        self.situational_pictures_red: list[SituationalPicture] = []
        self.monostatic_radar_detections_blue: list[MonostaticRadarDetection] = []
        self.pcl_detections_blue: list[PclDetection] = []
        self.pet_detections_blue: list[PetDetection] = []
        self.monostatic_radar_detections_red: list[MonostaticRadarDetection] = []
        self.pcl_detections_red: list[PclDetection] = []
        self.pet_detections_red: list[PetDetection] = []
        self.snapshots: list[Snapshot] = []
        self.events: list[Event] = []

    def register_simulator(self, simulator: Simulator):
        self.simulator = simulator

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
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        is_blue: bool,
    ):
        for det in active_radar_detections:
            if is_blue:
                self.monostatic_radar_detections_blue.append(det)
            else:
                self.monostatic_radar_detections_red.append(det)
        for det in pcl_detections:
            if is_blue:
                self.pcl_detections_blue.append(det)
            else:
                self.pcl_detections_red.append(det)
        for det in pet_detections:
            if is_blue:
                self.pet_detections_blue.append(det)
            else:
                self.pet_detections_red.append(det)

    def on_events(self, events: list[Event]):
        self.events.extend(events)

    def on_end(self):
        pass


class LogLoader:
    """Load JSON file written by FileLogger."""

    def __init__(self, path: str):
        with open(path, "r") as file:
            self._data = json.load(file)
        self._load_snapshots()
        self._load_sensors(is_blue=True)
        self._load_sensors(is_blue=False)
        self._load_target_ground_truth(is_blue=False)
        self._load_detections(is_blue=True)
        self._load_detections(is_blue=False)
        self._load_events()
        self._load_situational_pictures()

    @property
    def blue_monostatic_radars(self) -> list[MonostaticSensor]:
        return list(self._blue_monostatic_sensors.values())

    @property
    def blue_pcl_sensors(self) -> list[PclSensor]:
        return list(self._blue_pcl_sensors.values())

    @property
    def red_monostatic_radars(self) -> list[MonostaticSensor]:
        return list(self._red_monostatic_sensors.values())

    @property
    def red_pcl_sensors(self) -> list[PclSensor]:
        return list(self._red_pcl_sensors.values())

    @property
    def blue_monostatic_radar_detections(self) -> list[MonostaticRadarDetection]:
        return self._blue_monostatic_radar_detections

    @property
    def blue_pcl_detections(self) -> list[PclDetection]:
        return self._blue_pcl_detections

    @property
    def blue_situational_pictures(self) -> list[SituationalPicture]:
        return self._blue_situational_pictures

    @property
    def red_situational_pictures(self) -> list[SituationalPicture]:
        return self._red_situational_pictures

    @property
    def red_monostatic_radar_detections(self) -> list[MonostaticRadarDetection]:
        return self._red_monostatic_radar_detections

    @property
    def red_pcl_detections(self) -> list[PclDetection]:
        return self._red_pcl_detections

    @property
    def red_target_ground_truth(self) -> dict[int, GroundTruthPath]:
        return self._red_target_ground_truth

    @property
    def track_init_events(self) -> list[TrackInitEvent]:
        return [e for e in self._events if type(e) is TrackInitEvent]

    @property
    def shots(self) -> list[Shot]:
        return [e for e in self._events if isinstance(e, Shot)]

    @property
    def text_events(self) -> list[TextEvent]:
        return [e for e in self._events if isinstance(e, TextEvent)]

    @property
    def kill_events(self) -> list[KillEvent]:
        return [e for e in self._events if type(e) is KillEvent]

    @property
    def t_max(self) -> datetime.datetime:
        return self._snapshots[-1].time

    def _load_snapshots(self):
        self._snapshots = [Snapshot.model_validate(d) for d in self._data["snapshots"]]

    def _load_situational_pictures(self):
        # Situational pictures are only logged if debugging is enabled to keep
        # output file small.
        if "situational_pictures" in self._data:
            self._blue_situational_pictures = [
                SituationalPicture.model_validate(p["situational_picture"])
                for p in self._data["situational_pictures"]
                if p["team"] == "blue"
            ]
            self._red_situational_pictures = [
                SituationalPicture.model_validate(p["situational_picture"])
                for p in self._data["situational_pictures"]
                if p["team"] == "red"
            ]

    def _load_sensors(self, is_blue: bool):
        monostatic_sensors: dict[int, MonostaticSensor] = {}
        pcl_sensors: dict[int, PclSensor] = {}
        for snapshot in self._snapshots:
            sensors = (
                snapshot.blue_monostatic_radars
                if is_blue
                else snapshot.red_monostatic_radars
            )
            for sensor in sensors:
                monostatic_sensors[sensor.id] = sensor
            sensors = snapshot.blue_pcl_sensors if is_blue else snapshot.red_pcl_sensors
            for sensor in snapshot.blue_pcl_sensors:
                pcl_sensors[sensor.id] = sensor
        if is_blue:
            self._blue_monostatic_sensors = monostatic_sensors
            self._blue_pcl_sensors = pcl_sensors
        else:
            self._red_monostatic_sensors = monostatic_sensors
            self._red_pcl_sensors = pcl_sensors

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
        # Monostatic detections.
        monostatic_detections = list(
            itertools.chain.from_iterable(
                [
                    det["active_radar_detections"]
                    for det in self._data["detections"]
                    if det["team"] == ("blue" if is_blue else "red")
                ]
            )
        )
        monostatic_detections = [
            MonostaticRadarDetection.model_validate(d) for d in monostatic_detections
        ]
        if is_blue:
            self._blue_monostatic_radar_detections = monostatic_detections
        else:
            self._red_monostatic_radar_detections = monostatic_detections

        # PCL detections.
        pcl_detections = list(
            itertools.chain.from_iterable(
                [
                    det["pcl_detections"]
                    for det in self._data["detections"]
                    if det["team"] == ("blue" if is_blue else "red")
                ]
            )
        )
        pcl_detections = [PclDetection.model_validate(d) for d in pcl_detections]

        if is_blue:
            self._blue_pcl_detections = pcl_detections
        else:
            self._red_pcl_detections = pcl_detections

    def _load_events(self):
        self._events = TypeAdapter(
            list[TrackInitEvent | Shot | KillEvent | TextEvent]
        ).validate_python(self._data["events"])


class CompositeSimulationListener(AbstractSimulationListener):
    def __init__(self, listeners: list[AbstractSimulationListener]):
        self._listeners = listeners

    def register_simulator(self, simulator: Simulator):
        for listener in self._listeners:
            listener.register_simulator(simulator)

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
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        is_blue: bool,
    ):
        for listener in self._listeners:
            listener.on_detections(
                active_radar_detections,
                pcl_detections,
                pet_detections,
                is_blue,
            )

    def on_end(self):
        for listener in self._listeners:
            listener.on_end()

    def on_events(self, events: list[Event]):
        for listener in self._listeners:
            listener.on_events(events)


class FilterSimulationListener(AbstractSimulationListener):
    def __init__(
        self,
        listener: AbstractSimulationListener,
        forward_snapshots: bool,
        forward_situational_pictures: bool,
        forward_detections: bool,
        forward_events: bool,
    ):
        self._listener = listener
        self._forward_snapshots = forward_snapshots
        self._forward_detections = forward_detections
        self._forward_situational_pictures = forward_situational_pictures
        self._forward_events = forward_events

    def register_simulator(self, simulator: Simulator):
        pass

    def on_snapshot(self, snapshot):
        if self._forward_snapshots:
            self._listener.on_snapshot(snapshot)

    def on_situational_picture(self, situational_picture, is_blue):
        if self._forward_situational_pictures:
            self._listener.on_situational_picture(situational_picture, is_blue)

    def on_detections(
        self,
        active_radar_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        is_blue,
    ):
        if self._forward_detections:
            self._listener.on_detections(
                active_radar_detections,
                pcl_detections,
                pet_detections,
                is_blue,
            )

    def on_events(self, events: list[Event]):
        if self._forward_events:
            self._listener.on_events(events)

    def on_end(self):
        self._listener.on_end()


class SituationalPictureBuffer(AbstractSimulationListener):
    def __init__(self):
        self._red_ground_truth_history: dict[
            int, list[tuple[datetime.datetime, Target]]
        ] = {}
        self._blue_ground_truth_history: dict[
            int, list[tuple[datetime.datetime, Target]]
        ] = {}
        self._situational_picture_blue = SituationalPicture(
            time=datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc),
            friendly_pet_receivers=[],
            friendly_radars=[],
            friendly_targets=[],
            enemy_targets=[],
        )
        self._situational_picture_red = SituationalPicture(
            time=datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc),
            friendly_pet_receivers=[],
            friendly_radars=[],
            friendly_targets=[],
            enemy_targets=[],
        )
        self._events: list[Event] = []
        self._has_completed = False
        self._death_times: dict[int, datetime.datetime] = {}
        """
        Time of death for target IDs.
        If a target ID has no entry, the corresponding target is still alive.
        """
        self._geojson_blue: dict[str, GeoJSONFeature] = {}
        self._geojson_red: dict[str, GeoJSONFeature] = {}

    def register_simulator(self, simulator: Simulator):
        self._simulator = simulator
        self._geojson_blue = simulator._blue_controller.get_geojson()
        self._geojson_red = simulator._red_controller.get_geojson()

    def on_snapshot(self, snapshot: Snapshot):
        for target in snapshot.red_targets:
            traj = self._red_ground_truth_history.get(target.id, [])
            traj.append((snapshot.time, target))
            self._red_ground_truth_history[target.id] = traj
        for target in snapshot.blue_targets:
            traj = self._blue_ground_truth_history.get(target.id, [])
            traj.append((snapshot.time, target))
            self._blue_ground_truth_history[target.id] = traj

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
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        is_blue: bool,
    ):
        pass

    def on_events(self, events: list[Event]):
        kill_events = [e for e in events if isinstance(e, KillEvent)]
        for e in kill_events:
            self._death_times[e.target_id] = e.time
        self._events.extend(events)

    def on_end(self):
        self._has_completed = True

    def get_ground_truth_trajectories(self, is_blue: bool) -> list[Trajectory]:
        history = (
            self._blue_ground_truth_history.copy().values()
            if is_blue
            else self._red_ground_truth_history.copy().values()
        )
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
                if target.id in self._death_times:
                    # Do not return dead targets.
                    break
                times.append(time)
                lats.append(target.lat)
                lons.append(target.lon)
                alts.append(target.alt)
                vxs.append(target.velocity.vx)
                vys.append(target.velocity.vy)
                vzs.append(target.velocity.vz)
            if len(times) >= 2:
                trajectories.append(
                    Trajectory(
                        target_id=target.id,
                        target_sidc=target.sidc,
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

    def get_simulator(self) -> Simulator:
        return self._simulator

    def has_comleted(self) -> bool:
        return self._has_completed

    def get_events(self) -> list[Event]:
        return self._events

    def get_geojson(self, is_blue: bool) -> dict[str, GeoJSONFeature]:
        return self._geojson_blue if is_blue else self._geojson_red
