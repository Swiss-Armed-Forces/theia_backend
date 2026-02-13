import abc
import datetime
import itertools
import time
from theia.types import (
    Controller,
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
        """
        self._blue_controller = blue_controller
        self._red_controller = red_controller
        self._t = start_time
        self._dt = time_step
        self._minimum_time_per_step = min_time_per_step.seconds
        """Minimum amount of time to spend on an iteration."""
        self._termination_criterion = termination_criterion

        self._blue_receivers: list[Receiver] = []
        self._blue_transmitters: list[Transmitter] = []
        self._blue_targets: list[Target] = []
        self._red_receivers: list[Receiver] = []
        self._red_transmitters: list[Transmitter] = []
        self._red_targets: list[Target] = []

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

    def advance(self):
        start_time = time.time()

        # Build situational picture.
        blue_situational_picture = self.get_situational_picture_blue()
        red_situational_picture = self.get_situational_picture_red()

        # Update world according to behaviour.
        self._blue_receivers = self._blue_controller.get_receivers(
            blue_situational_picture
        )
        self._blue_transmitters = self._blue_controller.get_transmitters(
            blue_situational_picture
        )
        self._blue_targets = self._blue_controller.get_targets(blue_situational_picture)
        self._red_receivers = self._red_controller.get_receivers(
            red_situational_picture
        )
        self._red_transmitters = self._red_controller.get_transmitters(
            red_situational_picture
        )
        self._red_targets = self._red_controller.get_targets(red_situational_picture)

        # TODO: Detect.
        # TODO: Track.
        # TODO: Implement the update.

        stop_time = time.time()

        # Sleep the remaining time for an update.
        time_step_on_update = stop_time - start_time
        time.sleep(max(0, self._minimum_time_per_step - time_step_on_update))
