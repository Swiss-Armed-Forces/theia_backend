import datetime
import unittest

import numpy as np

from theia.detection.pcl import PclDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.pcl_sensor_controller import PclSensorController
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.pseudo_tracker import PseudoTracker
from theia.simulation.simulator import (
    AbstractSimulationListener,
    Simulator,
    TimeCriterion,
)
from theia.test_data import load_pcl_example
from theia.types import SituationalPicture


class PseudoTrackerTest(unittest.TestCase, AbstractSimulationListener):
    def test_pcl_only_track_init(self):
        sensors, trajectories, grid = load_pcl_example(rotation_time=1.0)
        self._trajectory = trajectories[0]

        scripted_target_controller = ControllerGroup(
            [WaypointTargetController.from_trajectory(t) for t in trajectories]
        )
        pcl_detector = PclDetector()

        start_time = min([t.times[0] for t in trajectories])
        stop_time = max([t.times[-1] for t in trajectories])

        time_step = datetime.timedelta(seconds=1)

        rng = np.random.Generator(np.random.PCG64(seed=4054080))

        simulator = Simulator(
            pcl_detector=pcl_detector,
            blue_controller=ControllerGroup(
                [PclSensorController(sensor) for sensor in sensors]
            ),
            red_controller=scripted_target_controller,
            blue_tracker=PseudoTracker(
                removal_patience=30,
                rng=rng,
                start_time=start_time,
            ),
            start_time=start_time,
            time_step=time_step,
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(stop_time),
            rng=rng,
            listener=self,
            simulate_clutter=False,
        )

        while simulator.advance():
            pass

    def register_simulator(self, simulator):
        self._first_detection = True

    def on_snapshot(self, snapshot):
        pass

    def on_situational_picture(
        self,
        situational_picture: SituationalPicture,
        is_blue: bool,
    ):
        if not is_blue:
            return
        # Check that the first tracked target is only when the track init region
        # is entered.
        if len(situational_picture.enemy_targets) > 0 and self._first_detection:
            self.assertLessEqual(
                abs(
                    situational_picture.time.timestamp()
                    - self._trajectory.times[1].timestamp()
                ),
                1,
            )
            self._first_detection = False

        if not self._first_detection:
            self.assertEqual(len(situational_picture.enemy_targets), 1)

    def on_detections(self, active_radar_detections, pcl_detections, is_blue):
        self.assertEqual(len(active_radar_detections), 0)
        # print(
        #     len(pcl_detections),
        #     pcl_detections[0].time if len(pcl_detections) > 0 else "",
        # )

    def on_end(self):
        pass


if __name__ == "__main__":
    unittest.main()
