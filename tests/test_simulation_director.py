import datetime
import json
import pathlib
import tempfile
import threading
import types
import unittest

import numpy as np
from fastapi.testclient import TestClient

from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.damage_model import UniformDamageModel
from theia.simulation.server import create_app
from theia.simulation.simulation_director import SimulationDirector
from theia.simulation.simulator import NeverCriterion, Simulator
from theia.simulation.theia_logging import FileLogger
from theia.simulation.trackers.tracking import DummyTracker
from theia.terrain import DummyTerrain

T0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)


class _CountingFileLogger(FileLogger):
    def __init__(self, path: str):
        super().__init__(path=path)
        self.n_on_end = 0
        self.n_snapshots_seen = 0

    def on_snapshot(self, snapshot):
        super().on_snapshot(snapshot)
        self.n_snapshots_seen += 1

    def on_end(self):
        super().on_end()
        self.n_on_end += 1


def build_simulator(path: str) -> tuple[Simulator, _CountingFileLogger]:
    """A simulator that never terminates by itself."""
    terrain = DummyTerrain(has_los=True)
    rng = np.random.Generator(np.random.PCG64(seed=1))
    logger = _CountingFileLogger(path)
    simulator = Simulator(
        pcl_detector=PclDetector(),
        pet_detector=PetDetector(terrain_model=terrain),
        blue_controller=ControllerGroup(controllers=[]),
        red_controller=ControllerGroup(controllers=[]),
        blue_tracker=DummyTracker(),
        red_tracker=DummyTracker(),
        start_time=T0,
        time_step=datetime.timedelta(seconds=1),
        min_time_per_step=datetime.timedelta(seconds=0),
        termination_criterion=NeverCriterion(),
        rng=rng,
        listener=logger,
        terrain_model=terrain,
        damage_model=UniformDamageModel(1.0, rng),
        simulate_clutter=False,
    )
    return simulator, logger


class _TempDirTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._dir.name) / "log.json"
        self.simulator, self.logger = build_simulator(str(self.path))

    def tearDown(self):
        self._dir.cleanup()

    def assert_log_written(self):
        self.assertEqual(self.logger.n_on_end, 1)
        with open(self.path) as file:
            data = json.load(file)
        self.assertEqual(len(data["snapshots"]), self.logger.n_snapshots_seen)


class SimulatorStopTest(_TempDirTest):
    def test_request_stop_ends_at_next_iteration(self):
        self.assertTrue(self.simulator.advance())
        self.assertTrue(self.simulator.advance())
        self.assertFalse(self.simulator.has_ended())
        self.simulator.request_stop()
        self.assertFalse(self.simulator.advance())
        self.assertTrue(self.simulator.has_ended())
        self.assert_log_written()
        self.assertEqual(self.logger.n_snapshots_seen, 2)

    def test_listener_notified_once(self):
        self.simulator.request_stop()
        for _ in range(3):
            self.assertFalse(self.simulator.advance())
        self.assertEqual(self.logger.n_on_end, 1)


class DirectorStopTest(_TempDirTest):
    def test_stop_while_running(self):
        director = SimulationDirector(self.simulator)
        # Let it run for a while.
        while self.logger.n_snapshots_seen < 5:
            pass
        director.stop()
        self.assertTrue(director.is_stopped())
        self.assertTrue(self.simulator.has_ended())
        self.assert_log_written()
        self.assertFalse(director._sim_thread.is_alive())

    def test_stop_while_paused(self):
        director = SimulationDirector(self.simulator)
        while self.logger.n_snapshots_seen < 5:
            pass
        director.pause()
        director._sim_thread.join()
        self.assertFalse(self.simulator.has_ended())
        director.stop()
        self.assertTrue(self.simulator.has_ended())
        self.assert_log_written()

    def test_resume_after_stop_does_nothing(self):
        director = SimulationDirector(self.simulator)
        director.stop()
        thread = director._sim_thread
        director.resume()
        self.assertIs(director._sim_thread, thread)
        self.assertTrue(director.is_paused())
        self.assert_log_written()

    def test_concurrent_stops(self):
        director = SimulationDirector(self.simulator)
        threads = [threading.Thread(target=director.stop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assert_log_written()


class StopEndpointTest(_TempDirTest):
    def test_stop_endpoint(self):
        director = SimulationDirector(self.simulator)
        client = TestClient(create_app(types.SimpleNamespace(), director))
        self.assertIs(client.get("/is_stopped").json(), False)
        response = client.post("/stop")
        self.assertEqual(response.status_code, 200, response.text)
        # The log is written once the request returns.
        self.assert_log_written()
        self.assertIs(client.get("/is_stopped").json(), True)
        self.assertIs(client.get("/is_paused").json(), True)


if __name__ == "__main__":
    unittest.main()
