import abc
import datetime
import itertools
import time

import numpy as np
from theia.detection.active import calculate_monostatic_detection
from theia.radar_equation import calculate_maximum_monostatic_range
from theia.simulation.logging import AbstractSimulationLogger
from theia.types import (
    MonostaticRadarDetection,
    Controller,
    MonostaticRadarMeasurementModel,
    Radar,
    Receiver,
    SituationalPicture,
    Snapshot,
    Target,
    Transmitter,
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
        blue_controller: Controller,
        red_controller: Controller,
        start_time: datetime.datetime,
        time_step: datetime.timedelta,
        min_time_per_step: datetime.timedelta,
        termination_criterion: TerminationCriterion,
        seed: int,
        logger: AbstractSimulationLogger,
        simulate_clutter: bool = True,
    ):
        """
        Parameters
        ----------
        blue_controller: Controller
            Handles all blue behaviour
        red_controller: Controller
            Handles all red behaviour
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
        logger: AbstractSimulationLogger
            Logger for intermediate results
        simulate_clutter: bool, default True
            Whether to simulate clutter detections
        """
        self._blue_controller = blue_controller
        self._red_controller = red_controller
        self._t = start_time
        self._dt = time_step
        self._minimum_time_per_step = min_time_per_step.seconds
        """Minimum amount of time to spend on an iteration."""
        self._termination_criterion = termination_criterion
        self._rng = np.random.Generator(np.random.PCG64(seed=seed))
        self._logger = logger
        self._simulate_clutter = simulate_clutter

        self._blue_receivers: list[Receiver] = []
        self._blue_transmitters: list[Transmitter] = []
        self._blue_targets: list[Target] = []
        self._red_receivers: list[Receiver] = []
        self._red_transmitters: list[Transmitter] = []
        self._red_targets: list[Target] = []
        self._active_detection_id = 0
        self._time_of_last_active_detection: dict[int, datetime.datetime] = {}
        """Time of latest detection for each active radar receiver ID."""

    def get_situational_picture_blue(self) -> SituationalPicture:
        radars = []
        for rx, tx in itertools.product(self._blue_receivers, self._blue_transmitters):
            radars.append(Radar(receiver=rx, transmitter=tx))
        return SituationalPicture(
            time=self._t,
            friendly_radars=radars,
            friendly_targets=self._blue_targets,
        )

    def get_situational_picture_red(self) -> SituationalPicture:
        radars = []
        for rx, tx in itertools.product(self._red_receivers, self._red_transmitters):
            radars.append(Radar(receiver=rx, transmitter=tx))
        return SituationalPicture(
            time=self._t,
            friendly_radars=radars,
            friendly_targets=self._red_targets,
        )

    def take_snapshot(self) -> Snapshot:
        return Snapshot(
            time=self._t,
            blue_transmitters=self._blue_transmitters,
            blue_receivers=self._blue_receivers,
            blue_targets=self._blue_targets,
            red_transmitters=self._red_transmitters,
            red_receivers=self._red_receivers,
            red_targets=self._red_targets,
        )

    @property
    def _blue_active_radars(self) -> list[Radar]:
        return [
            Radar(transmitter=tx, receiver=rx)
            for rx, tx in itertools.product(
                self._blue_receivers,
                self._blue_transmitters,
            )
            if rx.point == tx.point
        ]

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
        for radar in self._blue_active_radars:
            error_model = MonostaticRadarMeasurementModel(radar=radar)
            max_range = calculate_maximum_monostatic_range(radar)
            time_of_last_detection = self._time_of_last_active_detection.get(
                radar.receiver.id,
                datetime.datetime(year=1900, month=1, day=1, tzinfo=self._t.tzinfo),
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
                    error_model=error_model,
                )
                if det is not None:
                    det.time = self._t
                    det.detection_id = self._active_detection_id
                    self._active_detection_id += 1
                    detections.append(det)
            self._time_of_last_active_detection[radar.receiver.id] = self._t
            # Simulate clutter.
            if self._simulate_clutter:
                clutter_detections = error_model.sample_clutter(self._rng, max_range)
                for d in clutter_detections:
                    d.detection_id = self._active_detection_id
                    self._active_detection_id += 1
                    d.time = self._t
                    detections.append(d)
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
            self._logger.end()
            return False

        # Build situational picture.
        blue_situational_picture = self.get_situational_picture_blue()
        red_situational_picture = self.get_situational_picture_red()

        # Update world according to behaviour informed by situational picture.
        self._blue_receivers = self._blue_controller.get_receivers(
            blue_situational_picture,
            self._dt,
        )
        self._blue_transmitters = self._blue_controller.get_transmitters(
            blue_situational_picture,
            self._dt,
        )
        self._blue_targets = self._blue_controller.get_targets(
            blue_situational_picture,
            self._dt,
        )
        self._red_receivers = self._red_controller.get_receivers(
            red_situational_picture,
            self._dt,
        )
        self._red_transmitters = self._red_controller.get_transmitters(
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

        # TODO: Implement PCL detections.
        # TODO: Implement PET detections.
        # TODO: Track.

        # Log.
        self._logger.log_snapshot(self.take_snapshot())
        self._logger.log_detections(blue_active_radar_detections, is_blue=True)

        stop_time = time.time()

        # Sleep the remaining time for an update.
        time_step_on_update = stop_time - start_time
        time.sleep(max(0, self._minimum_time_per_step - time_step_on_update))

        return True
