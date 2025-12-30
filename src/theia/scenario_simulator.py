import datetime
import itertools
import numpy as np

from theia.detection.active import get_rad_pd
from theia.types import ActiveRadarDetection, RadarSimulator, TargetSimulator


class ScenarioSimulator:
    def __init__(
        self,
        target_simulator: TargetSimulator,
        radar_simulator: RadarSimulator,
        rng: np.random.Generator,
    ):
        self.target_simulator = target_simulator
        self.radar_simulator = radar_simulator
        self.rng = rng

        self._start_timestamp = min(
            self.target_simulator.get_minimum_time(),
            self.radar_simulator.get_minimum_time(),
        ).timestamp()
        self._stop_timestamp = max(
            self.target_simulator.get_maximum_time(),
            self.radar_simulator.get_maximum_time(),
        ).timestamp()

    def _get_simulated_times(self) -> list[datetime.datetime]:
        all_radars = self.radar_simulator.get_all_radars()
        rotation_times = set([radar.rotation_time for radar in all_radars])
        timestamps = []
        for t in rotation_times:
            timestamps.extend(
                np.arange(self._start_timestamp, self._stop_timestamp, step=t)
            )
        timestamps = sorted(list(set(timestamps)))
        return [datetime.datetime.fromtimestamp(s) for s in timestamps]

    def simulate_active_radar_detections(self) -> list[ActiveRadarDetection]:
        detections: list[ActiveRadarDetection] = []
        detection_id = 0
        times = self._get_simulated_times()
        for t in times:
            radars = self.radar_simulator.get_radars(t)
            # Only keep radars whose rotation time is aligned with the timestamp.
            radars = [
                r
                for r in radars
                if np.isclose(
                    (t.timestamp() - self._start_timestamp) % r.rotation_time,
                    0,
                )
            ]
            targets = self.target_simulator.get_targets(t)
            for radar, target in itertools.product(radars, targets):
                probability_of_detection = get_rad_pd(radar, target)
                # TODO: Include probability of false alarm as well.
                if self.rng.random() <= probability_of_detection:
                    detections.append(
                        ActiveRadarDetection(
                            detection_id=detection_id,
                            time=t,
                            radar=radar,
                            target=target,
                        )
                    )
                    detection_id += 1
        return detections
