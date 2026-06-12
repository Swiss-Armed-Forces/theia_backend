import theia
from theia.types import (
    AbstractTracker,
    IdProvider,
    MonostaticRadarDetection,
    PclDetection,
    PetDetection,
)


class DummyTracker(AbstractTracker):
    def add_detections(
        self,
        monostatic_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        id_provider: IdProvider,  # needed to get unused track IDs
    ):
        pass

    def get_tracks(self) -> list[theia.types.Track]:
        return []
