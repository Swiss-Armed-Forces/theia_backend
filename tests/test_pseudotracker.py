import datetime
import unittest

import numpy as np

from theia.coordinates import CoordinateTransformations
from theia.detection.ecef_sampler import EcefDetectionSampler
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.pcl_sensor_controller import PclSensorController
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.damage_model import UniformDamageModel
from theia.simulation.simulator import (
    AbstractSimulationListener,
    Simulator,
    TimeCriterion,
)
from theia.simulation.trackers.pseudo_tracker import (
    PseudoTracker,
    SingleTargetEcefTracker,
)
from theia.terrain import SrtmTerrainModel
from theia.test_data import load_pcl_example
from theia.types import (
    AbstractEventListener,
    ConstantRcsModel,
    Event,
    Party,
    Point,
    SituationalPicture,
    TargetInfos,
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
                WaypointTargetController.from_trajectory(
                    t,
                    TargetInfos.FIXED_WING.with_party(Party.RED),
                )
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
            damage_model=UniformDamageModel(1.0, rng),
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

    def on_events(self, events: list[Event]):
        """Needed because this class is an AbstractSimulationListener."""
        pass


class SingleTargetEcefTrackerTest(unittest.TestCase):
    """
    Filter-level regression tests. Detections come from the real
    `EcefDetectionSampler`, so the sampler and the tracker are tested together.
    """

    T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    SCAN_PERIOD = datetime.timedelta(seconds=10)
    START = Point(lat=47.0, lon=8.0, alt=1000.0)

    def _truth_ecef(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        x0 = np.array(
            CoordinateTransformations.geodetic_to_cartesian(*self.START.as_tuple())
        )
        return x0 + velocity * dt

    def _detect(
        self,
        sampler: EcefDetectionSampler,
        truth_ecef: np.ndarray,
        sigma: float,
        time: datetime.datetime,
    ):
        lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(*truth_ecef)
        return sampler._sample_position(
            Point(lat=lat, lon=lon, alt=alt),
            (sigma, sigma, sigma),
            time,
        )

    def _make_tracker(self) -> SingleTargetEcefTracker:
        return SingleTargetEcefTracker(
            id=0,
            t0=self.T0,
            pos0=(0.0, 0.0, 0.0),
        )

    def test_first_update_takes_measurement_uncertainty(self):
        """
        Directly after the first detection, the track must be exactly as
        uncertain as that detection, not more certain.
        """
        sampler = EcefDetectionSampler(np.random.default_rng(0))
        tracker = self._make_tracker()
        sigma = 500.0
        detection = self._detect(
            sampler,
            self._truth_ecef(np.zeros(3), 0.0),
            sigma,
            self.T0 + datetime.timedelta(minutes=10),
        )
        tracker.add_detection(detection)

        state = tracker._track[-1]
        np.testing.assert_allclose(
            np.diag(state.covar)[[0, 2, 4]],
            [sigma**2] * 3,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            np.array(state.state_vector).flatten()[[0, 2, 4]],
            np.array(detection.state_vector).flatten(),
            atol=1.0,
        )

    def test_late_first_detection_does_not_imply_stationary_target(self):
        """
        A track started long after the tracker's t0 must not assume that the
        target has been sitting at its first detected position since t0.
        Otherwise, a fast target is believed to be (nearly) stationary.
        """
        velocity = np.array([300.0, 0.0, 0.0])  # [m / s]
        sigma = 50.0
        t_first = self.T0 + datetime.timedelta(seconds=11)
        speeds = []
        for seed in range(20):
            sampler = EcefDetectionSampler(np.random.default_rng(seed))
            tracker = self._make_tracker()
            for k in range(2):
                t = t_first + k * self.SCAN_PERIOD
                dt = (t - t_first).total_seconds()
                tracker.add_detection(
                    self._detect(sampler, self._truth_ecef(velocity, dt), sigma, t)
                )
            v_est = np.array(tracker._track[-1].state_vector).flatten()[[1, 3, 5]]
            speeds.append(np.linalg.norm(v_est))

        # Two detections 10 s apart with 50 m noise: velocity error ~7 m/s.
        self.assertAlmostEqual(np.mean(speeds), 300.0, delta=15.0)

    def test_static_target_estimate_is_consistent(self):
        """
        The track's error must be consistent with the covariance it reports
        (normalised estimation error squared, NEES, of ~1 per dimension).

        An overconfident filter takes each noisy detection at face value and
        makes up velocities from the noise, so that a static ground target
        appears to move by hundreds of meters between detections.
        """
        x_true = self._truth_ecef(np.zeros(3), 0.0)
        sigma = 1000.0
        n_runs = 100
        n_scans = 20
        normalised_squared_errors = []
        for seed in range(n_runs):
            sampler = EcefDetectionSampler(np.random.default_rng(seed))
            tracker = self._make_tracker()
            for k in range(n_scans):
                t = self.T0 + (k + 1) * self.SCAN_PERIOD
                tracker.add_detection(self._detect(sampler, x_true, sigma, t))
            state = tracker._track[-1]
            x_est = np.array(state.state_vector).flatten()[[0, 2, 4]]
            variances = np.diag(state.covar)[[0, 2, 4]]
            normalised_squared_errors.append(
                np.mean((x_est - x_true) ** 2 / variances)
            )

        self.assertGreater(np.mean(normalised_squared_errors), 0.5)
        self.assertLess(np.mean(normalised_squared_errors), 2.0)


if __name__ == "__main__":
    unittest.main()
