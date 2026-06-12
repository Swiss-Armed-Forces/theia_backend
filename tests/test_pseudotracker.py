import datetime
import unittest

import numpy as np

from theia.config import SIDC
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.pcl_sensor_controller import PclSensorController
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.trackers.pseudo_tracker import PseudoTracker
from theia.simulation.simulator import (
    AbstractSimulationListener,
    Simulator,
    TimeCriterion,
)
from theia.terrain import SrtmTerrainModel
from theia.test_data import load_pcl_example
from theia.types import (
    AbstractEventListener,
    ConstantRcsModel,
    Event,
    SituationalPicture,
    TrackInitEvent,
)


class PseudoTrackerTest(
    unittest.TestCase,
    AbstractSimulationListener,
    AbstractEventListener,
):
    def test_pcl_only_track_init(self):
        sensors, trajectories, grid = load_pcl_example(rotation_time=1.0)
        self._trajectory = trajectories[0]

        scripted_target_controller = ControllerGroup(
            [
                WaypointTargetController.from_trajectory(t, SIDC.RED_FIXED_WING)
                for t in trajectories
            ]
        )

        start_time = min([t.times[0] for t in trajectories])
        stop_time = max([t.times[-1] for t in trajectories])

        time_step = datetime.timedelta(seconds=1)

        rng = np.random.Generator(np.random.PCG64(seed=4054080))

        terrain = SrtmTerrainModel()

        simulator = Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=ControllerGroup(
                [
                    PclSensorController(
                        len(scripted_target_controller._controllers) + 2 * sensor.id,
                        len(scripted_target_controller._controllers)
                        + 2 * sensor.id
                        + 1,
                        sensor,
                        True,
                        ConstantRcsModel(rcs=1.0),
                        own_receiver=True,
                        own_transmitter=True,
                    )
                    for sensor in sensors
                ]
            ),
            red_controller=scripted_target_controller,
            blue_tracker=PseudoTracker(
                removal_patience=30,
                rng=rng,
                start_time=start_time,
            ),
            red_tracker=PseudoTracker(
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
            terrain_model=terrain,
        )
        simulator.register_event_listener(self)

        self._track_init_event_fired = False

        while simulator.advance():
            pass

        self.assertTrue(self._track_init_event_fired)

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
        # Check that the first track is created only when the track init region
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

    def on_detections(
        self, active_radar_detections, pcl_detections, pet_detections, is_blue
    ):
        self.assertEqual(len(active_radar_detections), 0)
        # print(
        #     len(pcl_detections),
        #     pcl_detections[0].time if len(pcl_detections) > 0 else "",
        # )

    def on_end(self):
        pass

    def on_event(self, event: Event):
        if isinstance(event, TrackInitEvent):
            self._track_init_event_fired = True


if __name__ == "__main__":
    unittest.main()
