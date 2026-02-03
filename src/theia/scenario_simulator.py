import datetime
import itertools
import numpy as np

from theia.config import ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD, RF_LOSS
from theia.detection.active import calculate_monostatic_detection, get_rad_pd
from theia.types import ActiveRadarDetection, RadarSimulator, TargetSimulator


class ScenarioSimulator:
    def __init__(
        self,
        target_simulator: TargetSimulator,
        radar_simulator: RadarSimulator,
        rng: np.random.Generator,
        doppler_shift_threshold: float = ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD,
        rf_loss: float = RF_LOSS,
    ):
        self.target_simulator = target_simulator
        self.radar_simulator = radar_simulator
        self.rng = rng
        self._doppler_shift_threshold = doppler_shift_threshold
        self._rf_loss = rf_loss

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
        rotation_times = set([radar.receiver.rotation_time for radar in all_radars])
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
                    (t.timestamp() - self._start_timestamp) % r.receiver.rotation_time,
                    0,
                )
            ]
            targets = self.target_simulator.get_targets(t)
            for radar, target in itertools.product(radars, targets):
                detection = calculate_monostatic_detection(
                    radar,
                    target,
                    self.rng,
                    doppler_shift_threshold_hz=self._doppler_shift_threshold,
                    rf_loss=self._rf_loss,
                )
                if detection is not None:
                    detection.detection_id = detection_id
                    detection.time = t
                    detections.append(detection)
                    detection_id += 1
        return detections
