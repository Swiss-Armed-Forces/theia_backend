import abc
import json

from theia.types import ActiveRadarDetection, SituationalPicture


class AbstractSimulationLogger(abc.ABC):
    @abc.abstractmethod
    def log_situational_picture(
        self,
        situational_picture: SituationalPicture,
        is_blue: bool,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_detections(
        self,
        active_radar_detections: list[ActiveRadarDetection],
        is_blue: bool,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def end(self):
        raise NotImplementedError()


class NoLogger(AbstractSimulationLogger):
    def log_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        pass

    def log_detections(
        self, active_radar_detections: list[ActiveRadarDetection], is_blue: bool
    ):
        pass

    def end(self):
        pass


class FileLogger(AbstractSimulationLogger):
    def __init__(self, path: str, override: bool = True):
        self._path = path
        self._override = override
        self.situational_pictures = []
        self.active_radar_detections = []

    def log_situational_picture(
        self,
        situational_picture: SituationalPicture,
        is_blue: bool,
    ):
        self.situational_pictures.append(
            {
                "team": "blue" if is_blue else "red",
                "situational_picture": situational_picture.model_dump(mode="json"),
            }
        )

    def log_detections(
        self,
        active_radar_detections: list[ActiveRadarDetection],
        is_blue: bool,
    ):
        self.active_radar_detections.append(
            {
                "team": "blue" if is_blue else "red",
                "active_radar_detections": [
                    d.model_dump(mode="json") for d in active_radar_detections
                ],
            }
        )

    def end(self):
        with open(self._path, "a" if not self._override else "w") as file:
            json.dump(
                {
                    # "situational_pictures": self.situational_pictures,
                    "active_radar_detections": self.active_radar_detections,
                },
                file,
            )


class PrintLogger(AbstractSimulationLogger):
    def log_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        print(f"Situational picture {'BLUE' if is_blue else 'RED'}:")
        print(situational_picture)

    def log_detections(
        self, active_radar_detections: list[ActiveRadarDetection], is_blue: bool
    ):
        print(f"Active radar detections {'BLUE' if is_blue else 'RED'}:")
        print(active_radar_detections)

    def end(self):
        pass


class InMemoryLogger(AbstractSimulationLogger):
    def __init__(self):
        self.situational_pictures = []
        self.active_radar_detections = []

    def log_situational_picture(
        self, situational_picture: SituationalPicture, is_blue: bool
    ):
        self.situational_pictures.append(
            {
                "is_blue": is_blue,
                "situational_picture": situational_picture.model_dump(mode="json"),
            }
        )

    def log_detections(
        self, active_radar_detections: list[ActiveRadarDetection], is_blue: bool
    ):
        for det in active_radar_detections:
            self.active_radar_detections.append(
                {
                    "is_blue": is_blue,
                    "detection": det.model_dump(mode="json"),
                }
            )

    def end(self):
        pass
