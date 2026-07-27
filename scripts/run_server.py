import datetime

import numpy as np
import uvicorn

from theia.detection.pcl import PclDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.damage_model import UniformDamageModel
from theia.simulation.server import create_app
from theia.simulation.simulation_director import SimulationDirector
from theia.simulation.simulator import (
    AbstractSimulationListener,
    NeverCriterion,
    Simulator,
)
from theia.simulation.theia_logging import SituationalPictureBuffer
from theia.simulation.trackers.pseudo_tracker import PseudoTracker
from theia.terrain import SrtmTerrainModel


class DummySimulationDirector(SimulationDirector, AbstractSimulationListener):
    """
    Replaces the simulation director in order to start a standalone server
    (e. g. for the scenario editor).
    """

    def __init__(self):
        t0 = datetime.datetime.fromtimestamp(0, tz=datetime.UTC)
        rng = np.random.default_rng(seed=847980)
        terrain = SrtmTerrainModel()
        simulator = Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PclDetector(),
            blue_controller=ControllerGroup(controllers=[]),
            red_controller=ControllerGroup(controllers=[]),
            blue_tracker=PseudoTracker(1, rng, t0),
            red_tracker=PseudoTracker(1, rng, t0),
            start_time=t0,
            time_step=datetime.timedelta(seconds=1),
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=NeverCriterion(),
            rng=rng,
            listener=self,
            terrain_model=terrain,
            damage_model=UniformDamageModel(p_kill=1.0, rng=rng),
        )
        super().__init__(simulator)

    def register_simulator(self, simulator):
        pass

    def on_snapshot(self, snapshot):
        pass

    def on_situational_picture(self, situational_picture, is_blue):
        pass

    def on_detections(
        self, active_radar_detections, pcl_detections, pet_detections, is_blue
    ):
        pass

    def on_end(self):
        pass

    def on_events(self, events):
        pass


if __name__ == "__main__":
    buffer = SituationalPictureBuffer()
    director = DummySimulationDirector()

    app = create_app(buffer, director)

    print("Start simulation...")

    uvicorn.run(app, host="127.0.0.1", port=8000)
