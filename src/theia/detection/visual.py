from dataclasses import dataclass

import numpy as np

from theia.config import UNKNOWN_ID, UNKNOWN_TIME
from theia.distance import line_of_sight_distance
from theia.terrain import AbstractTerrainModel
from theia.types import Target, VisualDetection, VisualSensor


@dataclass
class VisualDetector:
    terrain: AbstractTerrainModel

    def calculate_visual_detection(
        self,
        rng: np.random.Generator,
        sensor: VisualSensor,
        tgt: Target,
    ) -> VisualDetection | None:
        p_sensor = sensor.receiver.point
        p_target = tgt.point

        if (
            self.terrain.has_line_of_sight(p_sensor, p_target)
            and line_of_sight_distance(
                p_sensor.lat,
                p_sensor.lon,
                p_sensor.alt,
                p_target.lat,
                p_target.lon,
                p_target.alt,
            )
            <= sensor.detection_range
        ):
            return VisualDetection(
                detection_id=UNKNOWN_ID,
                time=UNKNOWN_TIME,
                sensor=sensor,
                target=tgt,
            )
