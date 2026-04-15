from __future__ import annotations
import abc
import cProfile
import datetime
import time

import numpy as np
from theia.detection.active import calculate_monostatic_detection
from theia.detection.pcl import PclDetector
from theia.radar_equation import calculate_maximum_monostatic_range
from theia.types import (
    AbstractTracker,
    MonostaticRadarDetection,
    Controller,
    PclDetection,
    Sensor,
    SituationalPicture,
    Snapshot,
    Target,
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


class Simulator:
    def __init__(
        self,
        pcl_detector: PclDetector,
        blue_controller: Controller,
        red_controller: Controller,
        blue_tracker: AbstractTracker,
        start_time: datetime.datetime,
        time_step: datetime.timedelta,
        min_time_per_step: datetime.timedelta,
        termination_criterion: TerminationCriterion,
        seed: int,
        listener: AbstractSimulationListener,
        simulate_clutter: bool = True,
    ):
        """
        Parameters
        ----------
        pcl_detector: PclDetector
            PCL Detector
        blue_controller: Controller
            Handles all blue behaviour
        red_controller: Controller
            Handles all red behaviour
        blue_tracker: AbstractTracker
            Tracker that generates the blue situational pictures
            (it detects red targets)
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
        seed: int
            Seed for the pseudo-random number generator (RNG)
        listener: AbstractSimulationListener
            Called when intermediate results are available
        simulate_clutter: bool, default True
            Whether to simulate clutter detections
        """
        self._pcl_detector = pcl_detector
        self._blue_controller = blue_controller
        self._red_controller = red_controller
        self._blue_tracker = blue_tracker
        self._t = start_time
        self._dt = time_step
        self._minimum_seconds_per_step: float = float(min_time_per_step.seconds)
        """Minimum amount of time to spend on an iteration."""
        self._termination_criterion = termination_criterion
        self._rng = np.random.Generator(np.random.PCG64(seed=seed))
        self._listener = listener
        self._simulate_clutter = simulate_clutter

        self._blue_monostatic_radars: list[Sensor] = []
        self._blue_pcl_sensors: list[Sensor] = []
        self._blue_targets: list[Target] = []
        self._red_monostatic_radars: list[Sensor] = []
        self._red_pcl_sensors: list[Sensor] = []
        self._red_targets: list[Target] = []
        self._detection_id = 0
        self._time_of_last_detection: dict[int, datetime.datetime] = {}
        """Time of latest detection for each sensor ID."""

        self._listener.register_simulator(self)

    def get_situational_picture_blue(self) -> SituationalPicture:
        return SituationalPicture(
            time=self._t,
            friendly_radars=self._blue_monostatic_radars,
            friendly_targets=self._blue_targets,
            enemy_targets=self._blue_tracker.get_tracks(),
        )

    def get_situational_picture_red(self) -> SituationalPicture:
        return SituationalPicture(
            time=self._t,
            friendly_radars=self._red_monostatic_radars,
            friendly_targets=self._red_targets,
            enemy_targets=[],
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

    def _calculate_blue_monostatic_detections(self) -> list[MonostaticRadarDetection]:
        """
        Calculate blue active radar detections at the current time step,
        i. e. the red targets detected by BLUE. Includes clutter.

        Notes
        -----
        The active detections are assumed to take place at a fixed period
        (the receiver's rotation time) all at once.
        No angular update is implemented.
        """
        detections: list[MonostaticRadarDetection] = []
        for radar in self._blue_monostatic_radars:
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
            for target in self._red_targets:
                det = calculate_monostatic_detection(
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

    def _calculate_blue_pcl_detections(self) -> list[PclDetection]:
        """
        Calculate blue PCL detections at the current time step,
        i. e. the red targets detected by BLUE. Includes clutter.

        Notes
        -----
        The detections are assumed to take place at a fixed period
        (the receiver's rotation time) all at once.
        No angular update is implemented.
        """
        detections: list[PclDetection] = []
        for sensor in self._blue_pcl_sensors:
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
            for target in self._red_targets:
                det = self._pcl_detector.calculate_pcl_detection(
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

        # Update the time stamp.
        self._t += self._dt

        # Detect RED targets.
        blue_active_radar_detections = self._calculate_blue_monostatic_detections()
        blue_pcl_detections = self._calculate_blue_pcl_detections()

        # TODO: Implement PCL detections.
        # TODO: Implement PET detections.

        # Track.
        self._blue_tracker.add_detections(blue_active_radar_detections)

        # Log.
        self._listener.on_snapshot(self.take_snapshot())
        self._listener.on_detections(
            blue_active_radar_detections + blue_pcl_detections,
            is_blue=True,
        )

        stop_time = time.time()

        # Sleep the remaining time for an update.
        time_step_on_update = stop_time - start_time
        time.sleep(max(0, self._minimum_seconds_per_step - time_step_on_update))

        return True

    def set_speedup(self, speedup_factor: float):
        self._minimum_seconds_per_step = self._dt.seconds / speedup_factor

    def get_speedup(self) -> float:
        return self._dt.seconds / self._minimum_seconds_per_step


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
