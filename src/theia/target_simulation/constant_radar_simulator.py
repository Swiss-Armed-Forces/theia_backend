import copy
import datetime
from typing import Iterable
from theia.types import Radar, RadarSimulator


class ConstantRadarSimulator(RadarSimulator):
    """Returns the same radars for each time step between on- and off-time."""

    def __init__(
        self,
        radars: list[Radar],
        on_time: datetime.datetime,
        off_time: datetime.datetime,
    ):
        self._radars = copy.deepcopy(radars)
        self._on_time = on_time
        self._off_time = off_time

    def get_all_radars(self) -> list[Radar]:
        """Return a list of all radars that might be active at some point."""
        return self._radars

    def get_radars(self, time: datetime.datetime) -> Iterable[Radar]:
        """Return an iterator over radars at the given time."""
        if time < self._on_time or time > self._off_time:
            return []
        else:
            return self._radars

    def get_minimum_time(self) -> datetime.datetime:
        return self._on_time

    def get_maximum_time(self) -> datetime.datetime:
        return self._off_time
