from dataclasses import dataclass

import numpy as np

from theia.config import UNKNOWN_ID, UNKNOWN_TIME
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

        if self.terrain.has_line_of_sight(p_sensor, p_target):
            return VisualDetection(
                detection_id=UNKNOWN_ID,
                time=UNKNOWN_TIME,
                sensor=sensor,
                target=tgt,
            )
