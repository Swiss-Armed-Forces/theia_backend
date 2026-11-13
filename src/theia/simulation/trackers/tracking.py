import theia
from theia.types import (
    AbstractTracker,
    IdProvider,
    IntelDetection,
    MonostaticRadarDetection,
    PclDetection,
    PetDetection,
    VisualDetection,
)


class DummyTracker(AbstractTracker):
    def add_detections(
        self,
        monostatic_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        visual_detections: list[VisualDetection],
        intel_detections: list[IntelDetection],
        id_provider: IdProvider,  # needed to get unused track IDs
    ):
        pass

    def get_tracks(self) -> list[theia.types.Track]:
        return []
