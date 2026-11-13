import datetime
import unittest
from unittest.mock import patch

import numpy as np
import pydantic

from theia.config import INTEL_DETECTION_SIGMA
from theia.coordinates import CoordinateTransformations
from theia.detection.ecef_sampler import EcefDetectionSampler
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.geojson_controller import GeoJsonController
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.damage_model import UniformDamageModel
from theia.simulation.server import create_app
from theia.simulation.simulator import KillEvent, Simulator, TimeCriterion
from theia.simulation.theia_logging import InMemoryLogger
from theia.simulation.trackers.pseudo_tracker import PseudoTracker
from theia.terrain import DummyTerrain
from theia.test_data import build_flores_monostatic_radar
from theia.types import (
    AbstractTracker,
    ConstantRcsModel,
    Controller,
    Event,
    IdProvider,
    IntelDetection,
    IntelSensor,
    MonostaticRadarDetection,
    MonostaticSensor,
    Party,
    Point,
    Sensor,
    SituationalPicture,
    Target,
    TargetInfos,
    Track,
    Trajectory,
)

T0 = datetime.datetime.fromtimestamp(1_000_000, datetime.UTC)
TARGET_ID = 7


def make_trajectory(
    t_start: datetime.datetime = T0,
    duration_s: float = 10.0,
) -> Trajectory:
    """Straight northbound flight at 100 m/s, sampled every 2 s."""
    n = 6
    dts = np.linspace(0.0, duration_s, n)
    v = 100.0
    lat0, lon0, alt0 = 47.0, 8.0, 3000.0
    # ~111 km per degree of latitude.
    lats = [lat0 + v * dt / 111_000.0 for dt in dts]
    return Trajectory(
        target_id=TARGET_ID,
        target_info=TargetInfos.FIXED_WING.with_party(Party.RED),
        times=[t_start + datetime.timedelta(seconds=float(dt)) for dt in dts],
        lats=lats,
        lons=[lon0] * n,
        alts=[alt0] * n,
        vxs=[0.0] * n,
        vys=[v] * n,
        vzs=[0.0] * n,
        cross_section_model=ConstantRcsModel(rcs=1.0),
    )


def get_dummy_situational_picture() -> SituationalPicture:
    return SituationalPicture(
        time=T0,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


class _IntelController(Controller):
    """Controller exposing a fixed list of intel sensors."""

    sensors: list[IntelSensor] = []

    def on_event(self, event: Event):
        pass

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        self.intel_sensors = self.sensors

    def add_controller(self, controller: Controller):
        pass


class _EmptyController(Controller):
    def on_event(self, event: Event):
        pass

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        pass


class _RecordingTracker(AbstractTracker):
    """Tracker stub recording the intel detections of each call."""

    intel_calls: list[list[IntelDetection]] = []

    def add_detections(
        self,
        monostatic_detections,
        pcl_detections,
        pet_detections,
        visual_detections,
        intel_detections,
        id_provider,
    ):
        self.intel_calls.append(list(intel_detections))

    def get_tracks(self) -> list[Track]:
        return []


class IntelSensorDetectTest(unittest.TestCase):
    def setUp(self):
        self._trajectory = make_trajectory()
        self._sensor = IntelSensor(id=1, trajectory=self._trajectory)

    def test_inside_span_returns_ground_truth(self):
        for t in [
            T0,
            T0 + datetime.timedelta(seconds=3.5),
            T0 + datetime.timedelta(seconds=10),
        ]:
            target = self._sensor.detect(t)
            self.assertIsNotNone(target)
            self.assertEqual(target, self._trajectory(t))
            self.assertEqual(target.id, TARGET_ID)

    def test_outside_span_returns_none(self):
        self.assertIsNone(self._sensor.detect(T0 - datetime.timedelta(seconds=1)))
        self.assertIsNone(self._sensor.detect(T0 + datetime.timedelta(seconds=11)))


class SensorUnionTest(unittest.TestCase):
    def test_intel_sensor_validates_and_round_trips(self):
        adapter = pydantic.TypeAdapter(list[Sensor])
        sensor = IntelSensor(id=3, trajectory=make_trajectory())
        validated = adapter.validate_python([sensor])
        self.assertIsInstance(validated[0], IntelSensor)
        round_tripped = adapter.validate_json(adapter.dump_json(validated))
        self.assertEqual(len(round_tripped), 1)
        self.assertIsInstance(round_tripped[0], IntelSensor)
        self.assertEqual(round_tripped[0].id, 3)
        self.assertEqual(round_tripped[0].trajectory, sensor.trajectory)
        self.assertEqual(round_tripped[0].sigma, sensor.sigma)

    def test_monostatic_sensor_still_validates_as_monostatic(self):
        adapter = pydantic.TypeAdapter(list[Sensor])
        radar = build_flores_monostatic_radar(
            Point(lat=47.0, lon=8.0, alt=500.0),
            sensor_id=1,
            rx_id=2,
            tx_id=3,
        )
        validated = adapter.validate_python([radar])
        self.assertEqual(len(validated), 1)
        self.assertIsInstance(validated[0], MonostaticSensor)
        round_tripped = adapter.validate_json(adapter.dump_json(validated))
        self.assertIsInstance(round_tripped[0], MonostaticSensor)
        self.assertEqual(len(round_tripped), 1)
        self.assertEqual(radar, round_tripped[0])

    def test_openapi_schema_contains_intel_sensor(self):
        schemas = create_app(None, None).openapi()["components"]["schemas"]
        self.assertIn("IntelSensor", schemas)


class IntelSensorForwardingTest(unittest.TestCase):
    def setUp(self):
        self._sensor = IntelSensor(id=1, trajectory=make_trajectory())
        self._child = _IntelController(sensors=[self._sensor])
        self._picture = get_dummy_situational_picture()
        self._dt = datetime.timedelta(seconds=1)

    def test_controller_group(self):
        other = IntelSensor(id=2, trajectory=make_trajectory())
        group = ControllerGroup([self._child, _IntelController(sensors=[other])])
        group.update(self._picture, self._dt)
        self.assertEqual(list(group.intel_sensors), [self._sensor, other])

    def test_living_controller(self):
        controller = LivingController(child=self._child, target_id=42)
        controller.update(self._picture, self._dt)
        self.assertEqual(controller.intel_sensors, [self._sensor])
        controller.on_event(KillEvent(id=1, time=T0, target_id=42))
        controller.update(self._picture, self._dt)
        self.assertEqual(controller.intel_sensors, [])

    def test_geojson_controller(self):
        controller = GeoJsonController(child=self._child, geojson_features={})
        controller.update(self._picture, self._dt)
        self.assertEqual(controller.intel_sensors, [self._sensor])


class SimulatorIntelDetectionsTest(unittest.TestCase):
    def test_blue_tracker_receives_intel_detections(self):
        rng = np.random.Generator(np.random.PCG64(seed=1))
        terrain = DummyTerrain(has_los=True)
        dt = datetime.timedelta(seconds=1)
        t_start = T0 - datetime.timedelta(seconds=3)
        # Trajectory spans [T0, T0 + 10 s]; simulate a few steps before and after.
        trajectory = make_trajectory()
        sensor = IntelSensor(id=1, trajectory=trajectory)
        blue_tracker = _RecordingTracker()
        red_tracker = _RecordingTracker()
        simulator = Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=_IntelController(sensors=[sensor]),
            red_controller=_EmptyController(),
            blue_tracker=blue_tracker,
            red_tracker=red_tracker,
            start_time=t_start,
            time_step=dt,
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(T0 + datetime.timedelta(seconds=15)),
            rng=rng,
            listener=InMemoryLogger(),
            simulate_clutter=False,
            terrain_model=terrain,
            damage_model=UniformDamageModel(1.0, rng),
        )

        detection_ids: list[int] = []
        n_steps_with_detections = 0
        while True:
            n_calls_before = len(blue_tracker.intel_calls)
            if not simulator.advance():
                break
            if len(blue_tracker.intel_calls) == n_calls_before:
                # No intel detections in current iteration.
                continue
            detections = blue_tracker.intel_calls[-1]
            t = simulator._t
            if trajectory.times[0] <= t <= trajectory.times[-1]:
                self.assertEqual(len(detections), 1)
                d = detections[0]
                self.assertEqual(d.time, t)
                self.assertEqual(d.target, trajectory(t))
                detection_ids.append(d.detection_id)
                n_steps_with_detections += 1
            else:
                self.assertEqual(detections, [])

        self.assertEqual(n_steps_with_detections, 11)
        self.assertEqual(len(set(detection_ids)), len(detection_ids))
        self.assertGreater(len(red_tracker.intel_calls), 0)
        for detections in red_tracker.intel_calls:
            self.assertEqual(detections, [])


class PseudoTrackerIntelTest(unittest.TestCase):
    def _make_tracker(self) -> PseudoTracker:
        return PseudoTracker(
            removal_patience=5,
            rng=np.random.Generator(np.random.PCG64(seed=3)),
            start_time=T0,
        )

    def test_intel_only_creates_and_updates_track(self):
        trajectory = make_trajectory()
        tracker = self._make_tracker()
        id_provider = IdProvider()
        for i in range(11):
            t = T0 + datetime.timedelta(seconds=i)
            detection = IntelDetection(
                detection_id=i,
                time=t,
                target=trajectory(t),
                sigma=INTEL_DETECTION_SIGMA,
            )
            tracker.add_detections([], [], [], [], [detection], id_provider)
            if i == 0:
                # A single state does not yet make a track.
                self.assertEqual(tracker.get_tracks(), [])
            else:
                self.assertGreater(len(tracker.get_tracks()), 0)

        tracks = tracker.get_tracks()
        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual(track.target_info, trajectory.target_info)
        self.assertEqual(len(track.states), 11)

        t_last = trajectory.times[-1]
        truth = trajectory(t_last).point
        x, y, z = CoordinateTransformations.geodetic_to_cartesian(
            truth.lat, truth.lon, truth.alt
        )
        _, state = track.states[-1]
        error = np.linalg.norm(np.array([state[0], state[2], state[4]]) - [x, y, z])
        self.assertLess(error, 5 * INTEL_DETECTION_SIGMA)

    def test_intel_has_priority_over_monostatic(self):
        trajectory = make_trajectory()
        t = T0 + datetime.timedelta(seconds=2)
        ground_truth = trajectory(t)
        intel = IntelDetection(
            detection_id=1,
            time=t,
            target=ground_truth,
            sigma=INTEL_DETECTION_SIGMA,
        )
        # A monostatic detection of the same target ID with another `target`
        # payload, so that we can tell which one supplied it.
        other_target = Target(
            id=ground_truth.id,
            is_stationary=False,
            info=TargetInfos.UNKNOWN,
            point=ground_truth.point,
            cross_section_model=ground_truth.cross_section_model,
            velocity=ground_truth.velocity,
        )
        radar = build_flores_monostatic_radar(
            Point(lat=47.0, lon=7.9, alt=500.0), sensor_id=1, rx_id=2, tx_id=3
        )
        monostatic = MonostaticRadarDetection(
            detection_id=2,
            time=t,
            radar=radar,
            target=other_target,
            snr=20.0,
            target_range=10_000.0,
            elevation_angle=0.1,
            azimuth_angle=0.5,
            sigma_target_range=50.0,
            sigma_elevation=0.01,
            sigma_azimuth=0.01,
        )

        tracker = self._make_tracker()
        target_detections = tracker._preprocess_detections(
            [monostatic], [], [], [], [intel]
        )
        self.assertEqual(len(target_detections), 1)
        self.assertEqual(
            target_detections[0].target_info,
            ground_truth.info,
        )  # supplied by the intel detection
        self.assertEqual(target_detections[0].intel_detections, [intel])

        with patch.object(
            EcefDetectionSampler,
            "sample_monostatic",
            side_effect=AssertionError("monostatic must not be used"),
        ):
            tracker.add_detections([monostatic], [], [], [], [intel], IdProvider())
        sampler = EcefDetectionSampler(np.random.default_rng(0))
        detection = sampler.sample([monostatic], [], [], [], [intel], False)
        np.testing.assert_array_equal(
            detection.measurement_model.noise_covar,
            np.diag([INTEL_DETECTION_SIGMA] * 3),
        )


if __name__ == "__main__":
    unittest.main()
