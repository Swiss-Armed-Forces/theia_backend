from __future__ import annotations
import abc
import cProfile
import datetime
import itertools
import time

import numpy as np
from theia.detection.active import calculate_monostatic_detection
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.radar_equation import calculate_maximum_monostatic_range
from theia.terrain import AbstractTerrainModel
from theia.types import (
    AbstractEventListener,
    AbstractTracker,
    Event,
    IdProvider,
    MonostaticRadarDetection,
    Controller,
    MonostaticSensor,
    PclDetection,
    PclSensor,
    PetDetection,
    PetMeasurementModel,
    PetSensor,
    Receiver,
    SituationalPicture,
    Snapshot,
    Target,
    Trigger,
)


class TerminationCriterion(abc.ABC):
    @abc.abstractmethod
    def is_terminated(self, snapshot: Snapshot) -> bool:
        raise FileNotFoundError()


class TimeCriterion(TerminationCriterion):
    def __init__(self, end_time: datetime.datetime):
        self._end_time = end_time

    def is_terminated(self, snapshot: Snapshot) -> bool:
        return snapshot.time >= self._end_time


class Simulator(Trigger, AbstractEventListener):
    def __init__(
        self,
        pcl_detector: PclDetector,
        pet_detector: PetDetector,
        blue_controller: Controller,
        red_controller: Controller,
        blue_tracker: AbstractTracker,
        red_tracker: AbstractTracker,
        start_time: datetime.datetime,
        time_step: datetime.timedelta,
        min_time_per_step: datetime.timedelta,
        termination_criterion: TerminationCriterion,
        rng: np.random.Generator,
        listener: AbstractSimulationListener,
        terrain_model: AbstractTerrainModel,
        simulate_clutter: bool = True,
        id_provider: IdProvider = IdProvider(),
    ):
        """
        Parameters
        ----------
        pcl_detector: PclDetector
            PCL Detector
        pet_detector: PetDetector
            PET detector
        blue_controller: Controller
            Handles all blue behaviour
        red_controller: Controller
            Handles all red behaviour
        blue_tracker: AbstractTracker
            Tracker that generates the blue situational pictures
            (it detects red targets)
        red_tracker: AbstractTracker
            Tracker that generates the red situational pictures
            (it detects blue targets)
        start_time: datetime.datetime
            Initial time at the start of the simulation
        time_step: datetime.timedelta
            Duration of a single iteration [s]
        min_time_per_step: datetime.timedelta
            Minimum time to be spent for an iteration [s]
            Needed because we need to control the update rate for
            human interactivity.
        termination_criterion: TerminationCriterion
            Determines when the simulation stops.
        rng: np.random.Generator
            Pseudo-random number generator (RNG)
        listener: AbstractSimulationListener
            Called when intermediate results are available
        simulate_clutter: bool, default True
            Whether to simulate clutter detections
        """
        super().__init__()
        self._pcl_detector = pcl_detector
        self._pet_detector = pet_detector
        self._blue_controller = blue_controller
        self._red_controller = red_controller
        self._blue_tracker = blue_tracker
        self._red_tracker = red_tracker
        self._t = start_time
        self._dt = time_step
        self._minimum_seconds_per_step: float = float(min_time_per_step.seconds)
        """Minimum amount of time to spend on an iteration."""
        self._termination_criterion = termination_criterion
        self._rng = rng
        self._listener = listener
        self._simulate_clutter = simulate_clutter
        self._terrain_model = terrain_model

        self._blue_monostatic_radars: list[MonostaticSensor] = []
        self._blue_pcl_sensors: list[PclSensor] = []
        self._blue_targets: list[Target] = []
        self._red_monostatic_radars: list[MonostaticSensor] = []
        self._red_pcl_sensors: list[PclSensor] = []
        self._red_targets: list[Target] = []
        self._blue_pet_receivers: list[Receiver] = []
        self._red_pet_receivers: list[Receiver] = []
        self._detection_id = 0
        self._time_of_last_detection: dict[int, datetime.datetime] = {}
        """Time of latest detection for each sensor ID."""
        self._pet_sensor_ids: dict[tuple[int, int], int] = {}
        """IDs of PET sensors. Keys are (rx ID, tx ID) tuples."""
        self._id_provider = id_provider

        # Allow external reaction to simulation events.
        # Useful e. g. to expose simulation state to an API.
        self.set_listener(listener)

        # Broadcast events to controllers.
        self.register_event_listener(self._blue_controller)
        self.register_event_listener(self._red_controller)

        # Listen to events triggered by the controllers.
        self._blue_controller.register_event_listener(self)
        self._red_controller.register_event_listener(self)
        # Listen to events triggered by the trackers.
        self._blue_tracker.register_event_listener(self)
        self._red_tracker.register_event_listener(self)

    def _next_free_sensor_id(self) -> int:
        return (
            max(
                [r.id for r in self._blue_monostatic_radars]
                + [r.id for r in self._red_monostatic_radars]
                + [s.id for s in self._blue_pcl_sensors]
                + [s.id for s in self._red_pcl_sensors]
            )
            + 1
        )

    def set_listener(self, listener: AbstractSimulationListener):
        self._listener = listener
        self._listener.register_simulator(self)

    def get_situational_picture_blue(self) -> SituationalPicture:
        # for target in self._red_targets
        return SituationalPicture(
            time=self._t,
            friendly_pet_receivers=self._blue_pet_receivers,
            friendly_radars=self._blue_monostatic_radars + self._blue_pcl_sensors,
            friendly_targets=self._blue_targets,
            enemy_targets=self._blue_tracker.get_tracks(),
        )

    def get_situational_picture_red(self) -> SituationalPicture:
        return SituationalPicture(
            time=self._t,
            friendly_pet_receivers=self._red_pet_receivers,
            friendly_radars=self._red_monostatic_radars + self._red_monostatic_radars,
            friendly_targets=self._red_targets,
            enemy_targets=self._red_tracker.get_tracks(),
        )

    def take_snapshot(self) -> Snapshot:
        return Snapshot(
            time=self._t,
            blue_monostatic_radars=self._blue_monostatic_radars,
            blue_pcl_sensors=self._blue_pcl_sensors,
            blue_targets=self._blue_targets,
            red_monostatic_radars=self._red_monostatic_radars,
            red_pcl_sensors=self._red_pcl_sensors,
            red_targets=self._red_targets,
        )

    def _calculate_monostatic_detections(
        self,
        is_scanner_blue: bool,
    ) -> list[MonostaticRadarDetection]:
        """
        Calculate active radar detections at the current time step.
        Includes clutter.

        Parameters
        ----------
        is_scanner_blue: bool
            Whether the scanning part ("hunter") is blue; otherwise it is red

        Notes
        -----
        The active detections are assumed to take place at a fixed period
        (the receiver's rotation time) all at once.
        No angular update is implemented.
        """
        detections: list[MonostaticRadarDetection] = []
        sensors = (
            self._blue_monostatic_radars
            if is_scanner_blue
            else self._red_monostatic_radars
        )
        targets = self._red_targets if is_scanner_blue else self._blue_targets
        for radar in sensors:
            max_range = calculate_maximum_monostatic_range(radar)
            time_of_last_detection = self._time_of_last_detection.get(
                radar.id,
                datetime.datetime(
                    year=1900,
                    month=1,
                    day=1,
                    tzinfo=self._t.tzinfo,
                ),
            )
            if (
                self._t - time_of_last_detection
            ).seconds < radar.receiver.rotation_time:
                # No new detections.
                continue
            for target in targets:
                det = calculate_monostatic_detection(
                    self._terrain_model,
                    radar,
                    target,
                    rng=self._rng,
                    error_model=radar.error_model,
                )
                if det is not None:
                    det.time = self._t
                    det.detection_id = self._detection_id
                    self._detection_id += 1
                    detections.append(det)
            self._time_of_last_detection[radar.receiver.id] = self._t
            # Simulate clutter.
            if self._simulate_clutter:
                clutter_detections = radar.error_model.sample_clutter(
                    radar,
                    self._rng,
                    max_range,
                )
                for d in clutter_detections:
                    d.detection_id = self._detection_id
                    self._detection_id += 1
                    d.time = self._t
                    detections.append(d)
        return detections

    def _calculate_pcl_detections(self, is_scanner_blue: bool) -> list[PclDetection]:
        """
        Calculate PCL detections at the current time step.
        Does not include clutter.

        Parameters
        ----------
        is_scanner_blue: bool
            Whether the scanning part ("hunter") is blue; otherwise it is red

        Notes
        -----
        The detections are assumed to take place at a fixed period
        (the receiver's rotation time) all at once.
        No angular update is implemented.
        """
        detections: list[PclDetection] = []
        sensors = self._blue_pcl_sensors if is_scanner_blue else self._red_pcl_sensors
        targets = self._red_targets if is_scanner_blue else self._red_pcl_sensors
        for sensor in sensors:
            time_of_last_detection = self._time_of_last_detection.get(
                sensor.id,
                datetime.datetime(
                    year=1900,
                    month=1,
                    day=1,
                    tzinfo=self._t.tzinfo,
                ),
            )
            if (
                self._t - time_of_last_detection
            ).seconds < sensor.receiver.rotation_time:
                # No new detections.
                continue
            for target in targets:
                det = self._pcl_detector.calculate_pcl_detection(
                    self._rng,
                    sensor,
                    target,
                )
                if det is not None:
                    det.time = self._t
                    det.detection_id = self._detection_id
                    self._detection_id += 1
                    detections.append(det)
            self._time_of_last_detection[sensor.receiver.id] = self._t
            # Simulate clutter.
            if self._simulate_clutter:
                # TODO: Implement PCL clutter sampling.
                pass
        return detections

    def _calculate_pet_detections(self, is_scanner_blue: bool) -> list[PetDetection]:
        """
        Calculate PET detections at the current time step.
        Does not include clutter.

        Parameters
        ----------
        is_scanner_blue: bool
            Whether the scanning part ("hunter") is blue; otherwise it is red

        Notes
        -----
        The detections are assumed to take place at a fixed period
        (the receiver's rotation time) all at once.
        No angular update is implemented.
        """
        detections: list[PetDetection] = []

        sensors = self._blue_pet_sensors if is_scanner_blue else self._red_pet_sensors

        for sensor in sensors:
            time_of_last_detection = self._time_of_last_detection.get(
                sensor.id,
                datetime.datetime(
                    year=1900,
                    month=1,
                    day=1,
                    tzinfo=self._t.tzinfo,
                ),
            )
            if (
                self._t - time_of_last_detection
            ).seconds < sensor.receiver.rotation_time:
                # No new detections.
                continue

            det = self._pet_detector.calculate_pet_detection(
                sensor,
                sensor.target,
                self._rng,
            )
            if det is not None:
                det.time = self._t
                det.detection_id = self._detection_id
                self._detection_id += 1
                detections.append(det)
            self._time_of_last_detection[sensor.receiver.id] = self._t
            # Simulate clutter.
            if self._simulate_clutter:
                # TODO: Implement PET clutter sampling.
                pass

        return detections

    def advance(self) -> bool:
        """
        Advance the simulation by a single iteration.

        Returns
        -------
        bool
            Whether the iteration has run sucessfully
        """
        start_time = time.time()

        if self._termination_criterion.is_terminated(self.take_snapshot()):
            self._listener.on_end()
            return False

        # Build situational picture.
        blue_situational_picture = self.get_situational_picture_blue()
        red_situational_picture = self.get_situational_picture_red()

        self._listener.on_situational_picture(blue_situational_picture, True)
        self._listener.on_situational_picture(red_situational_picture, False)

        # Update world according to behaviour informed by situational picture.
        self._blue_monostatic_radars = self._blue_controller.get_monostatic_radars(
            blue_situational_picture,
            self._dt,
        )
        self._blue_pcl_sensors = self._blue_controller.get_pcl_sensors(
            blue_situational_picture,
            self._dt,
        )
        self._blue_targets = self._blue_controller.get_targets(
            blue_situational_picture,
            self._dt,
        )
        self._red_monostatic_radars = self._red_controller.get_monostatic_radars(
            red_situational_picture,
            self._dt,
        )
        self._red_pcl_sensors = self._red_controller.get_pcl_sensors(
            red_situational_picture,
            self._dt,
        )
        self._red_targets = self._red_controller.get_targets(
            red_situational_picture,
            self._dt,
        )

        self._blue_pet_receivers = self._blue_controller.get_pet_receivers(
            blue_situational_picture,
            self._dt,
        )
        self._blue_pet_sensors: list[PetSensor] = []
        for rx, target in itertools.product(
            self._blue_pet_receivers, self._red_targets
        ):
            if target.transmitter is not None:
                sensor = PetSensor(
                    id=self._pet_sensor_ids.get(
                        (rx.id, target.id),
                        self._next_free_sensor_id(),
                    ),
                    transmitter=target.transmitter,
                    receiver=rx,
                    target=target,
                    error_model=PetMeasurementModel(),
                )
                self._blue_pet_sensors.append(sensor)
                self._pet_sensor_ids[(rx.id, target.id)] = sensor.id

        self._red_pet_receivers = self._red_controller.get_pet_receivers(
            red_situational_picture,
            self._dt,
        )
        self._red_pet_sensors: list[PetSensor] = []
        for rx, target in itertools.product(
            self._red_pet_receivers, self._blue_targets
        ):
            if target.transmitter is not None:
                sensor = PetSensor(
                    id=self._pet_sensor_ids.get(
                        (rx.id, target.id),
                        self._next_free_sensor_id(),
                    ),
                    transmitter=target.transmitter,
                    receiver=rx,
                    target=target,
                    error_model=PetMeasurementModel(),
                )
                self._red_pet_sensors.append(sensor)
                self._pet_sensor_ids[(rx.id, target.id)] = sensor.id

        # Update the time stamp.
        self._t += self._dt

        # Detect RED targets.
        blue_active_radar_detections = self._calculate_monostatic_detections(
            is_scanner_blue=True
        )
        red_active_radar_detections = self._calculate_monostatic_detections(
            is_scanner_blue=False
        )
        blue_pcl_detections = self._calculate_pcl_detections(is_scanner_blue=True)
        red_pcl_detections = self._calculate_pcl_detections(is_scanner_blue=False)
        blue_pet_detections = self._calculate_pet_detections(is_scanner_blue=True)
        red_pet_detections = self._calculate_pet_detections(is_scanner_blue=False)

        # Track.
        self._blue_tracker.add_detections(
            blue_active_radar_detections,
            blue_pcl_detections,
            blue_pet_detections,
            self._id_provider,
        )
        self._red_tracker.add_detections(
            red_active_radar_detections,
            red_pcl_detections,
            red_pet_detections,
            self._id_provider,
        )

        # Log.
        self._listener.on_snapshot(self.take_snapshot())
        self._listener.on_detections(
            blue_active_radar_detections,
            blue_pcl_detections,
            blue_pet_detections,
            is_blue=True,
        )
        self._listener.on_detections(
            red_active_radar_detections,
            red_pcl_detections,
            red_pet_detections,
            is_blue=False,
        )

        stop_time = time.time()

        # Sleep the remaining time for an update.
        time_step_on_update = stop_time - start_time
        sleep_time = self._minimum_seconds_per_step - time_step_on_update
        if sleep_time > 0:
            time.sleep(sleep_time)

        return True

    def set_speedup(self, speedup_factor: float):
        self._minimum_seconds_per_step = self._dt.seconds / speedup_factor

    def get_speedup(self) -> float:
        return self._dt.seconds / max(1e-6, self._minimum_seconds_per_step)

    def on_event(self, event: Event):
        self._broadcast_event(event)


def run_simulation_until_completion(simulator: Simulator):
    is_running = True
    while is_running:
        is_running = simulator.advance()
    print("============================")
    print("DONE!")
    print("============================")


def run_with_profile(fn, profile_path: str):
    def wrapper(*args, **kwargs):
        cProfile.runctx(
            "fn(*args, **kwargs)",
            globals={"fn": fn},
            locals={"args": args, "kwargs": kwargs},
            filename=profile_path,
        )

    return wrapper


class AbstractSimulationListener(abc.ABC):
    @abc.abstractmethod
    def register_simulator(self, simulator: Simulator):
        raise NotImplementedError()

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
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        is_blue: bool,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def on_end(self):
        raise NotImplementedError()


def profile_simulation_until_completion(
    simulator: Simulator,
    profile_path: str = "sim_thread.prof",
):
    f = run_with_profile(run_simulation_until_completion, profile_path=profile_path)
    f(simulator)
