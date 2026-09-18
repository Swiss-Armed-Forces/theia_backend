import datetime
from pathlib import Path
import unittest

import numpy as np
from tqdm import tqdm

from theia.config import SIDC
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.data_loading import load_trajectory_file
from theia.data_loading_testing import load_pcl_reference_data
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.effectors import DirectFireEffector, IndirectFireEffector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.homing_effector import HomingSystem
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.static_gbad_controller import StaticGbadController
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.damage_model import UniformDamageModel
from theia.simulation.factories.single_target_single_effector import (
    SingleTargetSingleEffectorFactory,
)
from theia.simulation.theia_logging import FileLogger, InMemoryLogger
from theia.simulation.simulator import KillEvent, Simulator, TimeCriterion
from theia.simulation.trackers.tracking import DummyTracker
from theia.terrain import SrtmTerrainModel
from theia.test_data import get_uetliberg_radar
from theia.types import (
    AbstractEventListener,
    AbstractTracker,
    ConstantRcsModel,
    Controller,
    DirectShot,
    Event,
    IndirectShot,
    Point,
    SituationalPicture,
    Target,
    Track,
    Velocity,
)

FREQUENCY = 3_000  # Hz
POWER = 500_000  # W
DIAMETER = 4.0  # m
BANDWIDTH = 5  # MHz


class SimulatorTest(unittest.TestCase):
    def test_run_zueri_westbound(self):
        # Define monostatic radars.
        radar = get_uetliberg_radar()
        # Load targets.
        trajectories, _ = load_pcl_reference_data(
            f"{Path(__file__).resolve().parent}/test_data/pcl_detection"
        )
        scripted_target_controller = ControllerGroup(
            [
                WaypointTargetController.from_trajectory(t, SIDC.RED_FIXED_WING)
                for t in trajectories
            ]
        )

        # Build the simulator.
        start_time = min([t.times[0] for t in trajectories])
        stop_time = max([t.times[-1] for t in trajectories])

        logger = InMemoryLogger()

        terrain = SrtmTerrainModel()

        rng = np.random.Generator(np.random.PCG64(seed=4054080))

        simulator = Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=MonostaticRadarController(
                len(scripted_target_controller._controllers) + 1,
                radar,
                True,
                ConstantRcsModel(rcs=1.0),
            ),
            red_controller=scripted_target_controller,
            blue_tracker=DummyTracker(),
            red_tracker=DummyTracker(),
            start_time=start_time,
            time_step=datetime.timedelta(seconds=1),
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(stop_time),
            rng=rng,
            listener=logger,
            terrain_model=terrain,
            damage_model=UniformDamageModel(1.0, rng),
        )
        # Simulate until the end.
        while simulator.advance():
            pass

    def test_opensky(self):
        radar = get_uetliberg_radar()

        # Load trajectories from OpenSky.
        trajectories, _ = load_trajectory_file(
            f"{Path(__file__).resolve().parent}/../data/data_opensky_2022-06-27.csv"
        )

        scripted_target_controller = ControllerGroup(
            [
                WaypointTargetController.from_trajectory(t, SIDC.RED_FIXED_WING)
                for t in trajectories
            ]
        )

        # Build the simulator.
        start_time = min([t.times[0] for t in trajectories])
        stop_time = max([t.times[-1] for t in trajectories])

        logger = FileLogger("log_opensky.json")

        time_step = datetime.timedelta(seconds=1)

        terrain = SrtmTerrainModel()

        rng = np.random.Generator(np.random.PCG64(seed=4054080))

        simulator = Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=MonostaticRadarController(
                len(scripted_target_controller._controllers) + 1,
                radar,
                True,
                ConstantRcsModel(rcs=1.0),
            ),
            red_controller=scripted_target_controller,
            blue_tracker=DummyTracker(),
            red_tracker=DummyTracker(),
            start_time=start_time,
            time_step=time_step,
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(stop_time),
            rng=rng,
            listener=logger,
            terrain_model=terrain,
            damage_model=UniformDamageModel(1.0, rng),
        )
        n_iterations = (
            int(np.ceil((stop_time - start_time).seconds / time_step.seconds)) + 1
        )

        # Simulate until the end.
        for _ in tqdm(range(n_iterations)):
            is_success = simulator.advance()
            self.assertTrue(is_success)
        self.assertFalse(simulator.advance())


class SimulatorEffectorsTest(unittest.TestCase, AbstractEventListener):
    def setUp(self):
        self._already_killed = False

    def test_no_uncertainty(self):
        rng = np.random.Generator(np.random.PCG64(seed=4054080))
        terrain = SrtmTerrainModel()
        damage_model = UniformDamageModel(1.0, rng)  # Every shot kills.
        factory = SingleTargetSingleEffectorFactory(rng, terrain, False)
        simulator, buffer = factory.build_simulator(False, rng, terrain, damage_model)
        simulator.register_event_listener(self)

        while simulator.advance():
            pass

        self.assertTrue(self._already_killed)

    def test_with_uncertainty(self):
        rng = np.random.Generator(np.random.PCG64(seed=4054080))
        terrain = SrtmTerrainModel()
        damage_model = UniformDamageModel(1.0, rng)  # Every shot kills.
        factory = SingleTargetSingleEffectorFactory(rng, terrain, True)
        simulator, buffer = factory.build_simulator(False, rng, terrain, damage_model)
        simulator.register_event_listener(self)

        while simulator.advance():
            pass

        self.assertTrue(self._already_killed)

    def on_event(self, event: Event):
        # Ignore all but kill events.
        if not isinstance(event, KillEvent):
            return

        # Make sure that the correct target is killed.
        # Event with ID 1 is the shot, event with ID 2 the kill event.
        self.assertEqual(event.target_id, 1)
        self.assertEqual(event.id, 2)
        self.assertEqual(event.time, datetime.datetime.fromtimestamp(25, datetime.UTC))

        # Make sure that the kill event occurs only once.
        self.assertFalse(self._already_killed)
        self._already_killed = True


class _StationaryTargetController(Controller):
    """Minimal RED-side controller reporting one fixed ground-truth Target."""

    target_id: int
    point: Point

    def on_event(self, event: Event):
        pass

    def update(self, situational_picture: SituationalPicture, dt: datetime.timedelta):
        self.targets = [
            Target(
                id=self.target_id,
                is_stationary=True,
                sidc=SIDC.RED_FIXED_WING,
                point=self.point,
                cross_section_model=ConstantRcsModel(rcs=1.0),
                velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
            )
        ]


class _FixedTrackTracker(AbstractTracker):
    """Perfect-knowledge tracker stub: always reports one fixed Track."""

    def __init__(self, track: Track):
        super().__init__()
        self._track = track

    def add_detections(self, *args, **kwargs):
        pass

    def get_tracks(self) -> list[Track]:
        return [self._track]


class _EventRecorder(AbstractEventListener):
    def __init__(self):
        self.events: list[Event] = []

    def on_event(self, event: Event):
        self.events.append(event)


class SimulatorIndirectFireTest(unittest.TestCase):
    def test_launch_spawns_projectile_and_cadence_gates_relaunch(self):
        rng = np.random.Generator(np.random.PCG64(seed=1))
        terrain = SrtmTerrainModel()
        t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
        dt = datetime.timedelta(seconds=1)

        p_launcher = Point(
            lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
            lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
            alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
        )
        # Same point already exercised (successfully, LOS included) by
        # tests/test_static_direct_file_controller.py - known to be within
        # range/LOS of Uetliberg and covered by local terrain data.
        p_target = Point(lat=47.37348, lon=8.53707, alt=1000.0)
        x, y, z = CoordinateTransformations.geodetic_to_cartesian(
            p_target.lat, p_target.lon, p_target.alt
        )

        # Deliberately different from red_stub's ground-truth target_id below:
        # a track ID must never be assumed to equal a target ID (a track can
        # correspond to multiple targets, or to clutter).
        track = Track(
            id="6",
            sidc=SIDC.UNKNOWN,
            states=[
                (t0, np.array([x, 0.0, y, 0.0, z, 0.0])),
                (
                    t0 + datetime.timedelta(seconds=100),
                    np.array([x, 0.0, y, 0.0, z, 0.0]),
                ),
            ],
        )

        projectile_effector = DirectFireEffector(
            id=99,
            point=p_launcher,
            combat_range=5_000,
            n_attacks_left=1,
            name="",
            terrain=terrain,
            cadence=float("inf"),
        )
        effector = IndirectFireEffector(
            id=5,
            point=p_launcher,
            combat_range=100_000,
            n_attacks_left=5,
            name="",
            projectile=projectile_effector,
            projectile_speed=300.0,
            projectile_max_dist=50_000.0,
            projectile_sidc=SIDC.BLUE_MISSILE,
            projectile_rcs=ConstantRcsModel(rcs=0.1),
            # 0.5 => must wait 2s between shots (cadence lives on the
            # effector, not the controller - see tests/test_effectors.py).
            cadence=0.5,
        )
        launcher = StaticGbadController(
            target_id=4,
            sidc=SIDC.BLUE_AIR_DEFENCE,
            rcs=1.5,
            effector=effector,
            assigned_track_id="6",
        )
        blue_group = ControllerGroup([launcher])
        red_stub = _StationaryTargetController(target_id=1, point=p_target)

        simulator = Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=blue_group,
            red_controller=red_stub,
            blue_tracker=_FixedTrackTracker(track),
            red_tracker=DummyTracker(),
            start_time=t0,
            time_step=dt,
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(t0 + datetime.timedelta(seconds=1000)),
            rng=rng,
            listener=InMemoryLogger(),
            terrain_model=terrain,
            damage_model=UniformDamageModel(1.0, rng),
        )
        recorder = _EventRecorder()
        simulator.register_event_listener(recorder)

        def n_indirect_shots() -> int:
            return len([e for e in recorder.events if isinstance(e, IndirectShot)])

        # _execute_attacks matches firing_effectors against the *previous*
        # tick's ground-truth targets (deliberately, so shots and the
        # situational picture they were aimed from stay in sync - see the
        # "Fight before updating the world" comment in Simulator.advance).
        # Tick 0:
        # No targets yet, so no shots.
        self.assertTrue(simulator.advance())
        self.assertEqual(n_indirect_shots(), 0)

        # Tick 1:
        # A RED target existst, is within range and the launcher is not
        # on cooldown yet, so the launch fires immediately.
        self.assertTrue(simulator.advance())
        self.assertEqual(n_indirect_shots(), 1)

        spawned = [c for c in blue_group._controllers if c is not launcher]
        self.assertEqual(len(spawned), 1)
        living = spawned[0]
        self.assertIsInstance(living, LivingController)
        self.assertIsInstance(living.child, HomingSystem)
        # Must be the assigned track ID ("6"), not the ground-truth target ID
        # (1) the shot happened to resolve against.
        self.assertEqual(living.child.assigned_track_id, "6")

        # Tick 2:
        # cadence (2s cooldown) suppresses an immediate second launch.
        self.assertTrue(simulator.advance())
        self.assertEqual(n_indirect_shots(), 1)

        # Tick 3: the 2s cadence window has now elapsed - fires again.
        self.assertTrue(simulator.advance())
        self.assertEqual(n_indirect_shots(), 2)

    def test_relaunch_blocked_while_missile_in_flight_despite_fast_cadence(self):
        # Regression test: cadence alone must not determine the relaunch
        # rate for indirect fire. A launcher must not stack up multiple
        # projectiles against the same track just because its cadence
        # allows firing every tick - see IndirectFireEffector.max_in_flight
        # (default 1) / n_in_flight.
        rng = np.random.Generator(np.random.PCG64(seed=1))
        terrain = SrtmTerrainModel()
        t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
        dt = datetime.timedelta(seconds=1)

        p_launcher = Point(
            lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
            lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
            alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
        )
        # ~91km from the launcher: within the effector's own combat_range
        # (100km) so it can fire, but the projectile's combat_range (100m)
        # and max_dist (900m, i.e. 3s of fuel at 300 m/s) are both far too
        # small for it to ever reach the target - the missile stays "in
        # flight" for several ticks until it runs out of fuel.
        p_target = Point(lat=46.948056, lon=7.4475, alt=1000.0)
        x, y, z = CoordinateTransformations.geodetic_to_cartesian(
            p_target.lat, p_target.lon, p_target.alt
        )
        track = Track(
            id="6",
            sidc=SIDC.UNKNOWN,
            states=[
                (t0, np.array([x, 0.0, y, 0.0, z, 0.0])),
                (
                    t0 + datetime.timedelta(seconds=100),
                    np.array([x, 0.0, y, 0.0, z, 0.0]),
                ),
            ],
        )

        projectile_effector = DirectFireEffector(
            id=99,
            point=p_launcher,
            combat_range=100,
            n_attacks_left=1,
            name="",
            terrain=terrain,
            cadence=float("inf"),
        )
        effector = IndirectFireEffector(
            id=5,
            point=p_launcher,
            combat_range=100_000,
            n_attacks_left=5,
            name="",
            projectile=projectile_effector,
            projectile_speed=300.0,
            projectile_max_dist=900.0,
            projectile_sidc=SIDC.BLUE_MISSILE,
            projectile_rcs=ConstantRcsModel(rcs=0.1),
            # Effectively unlimited rate of fire - the in-flight cap
            # (max_in_flight, default 1) is the only thing that should
            # gate relaunches in this test.
            cadence=1000.0,
        )
        launcher = StaticGbadController(
            target_id=4,
            sidc=SIDC.BLUE_AIR_DEFENCE,
            rcs=1.5,
            effector=effector,
            assigned_track_id="6",
        )
        blue_group = ControllerGroup([launcher])
        red_stub = _StationaryTargetController(target_id=1, point=p_target)

        simulator = Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=blue_group,
            red_controller=red_stub,
            blue_tracker=_FixedTrackTracker(track),
            red_tracker=DummyTracker(),
            start_time=t0,
            time_step=dt,
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(t0 + datetime.timedelta(seconds=1000)),
            rng=rng,
            listener=InMemoryLogger(),
            terrain_model=terrain,
            damage_model=UniformDamageModel(1.0, rng),
        )
        recorder = _EventRecorder()
        simulator.register_event_listener(recorder)

        def n_indirect_shots() -> int:
            return len([e for e in recorder.events if isinstance(e, IndirectShot)])

        # Tick 0: no targets yet.
        self.assertTrue(simulator.advance())
        self.assertEqual(n_indirect_shots(), 0)

        # Tick 1: first launch.
        self.assertTrue(simulator.advance())
        self.assertEqual(n_indirect_shots(), 1)
        self.assertEqual(effector.n_in_flight, 1)

        # Ticks 2-4: despite cadence allowing a shot every tick, the
        # in-flight missile blocks any relaunch.
        for _ in range(3):
            self.assertTrue(simulator.advance())
            self.assertEqual(n_indirect_shots(), 1)
            self.assertEqual(effector.n_in_flight, 1)

        # Tick 5: the first missile has run out of fuel (self-destructed)
        # and freed up the in-flight slot, so a second launch fires.
        self.assertTrue(simulator.advance())
        self.assertEqual(n_indirect_shots(), 2)
        self.assertEqual(effector.n_in_flight, 1)


class TrackingErrorWastesAmmoTest(unittest.TestCase):
    """
    A track that does not correspond to any real, nearby target still gets
    fired at: the effector consumes ammo/cadence and the shot is broadcast,
    it just cannot result in a kill (direct fire) or a usefully-aimed
    projectile (indirect fire) - Theia should be able to represent
    ammunition wasted on tracking errors rather than silently withholding
    the shot whenever the track doesn't line up with the ground truth.
    """

    def _build_scenario(self):
        rng = np.random.Generator(np.random.PCG64(seed=2))
        terrain = SrtmTerrainModel()
        t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
        dt = datetime.timedelta(seconds=1)

        p_launcher = Point(
            lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
            lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
            alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
        )
        # Ground-truth target position.
        p_target = Point(lat=47.37348, lon=8.53707, alt=1000.0)
        # Track position ~556m away from the ground truth - beyond the
        # default 250m shot_association_tolerance, but still well within
        # combat range and LOS of the launcher (verified: both points have
        # clear LOS from Uetliberg).
        p_track = Point(lat=47.37348 + 0.005, lon=8.53707, alt=1000.0)
        x, y, z = CoordinateTransformations.geodetic_to_cartesian(
            p_track.lat, p_track.lon, p_track.alt
        )
        track = Track(
            id="9",
            sidc=SIDC.UNKNOWN,
            states=[
                (t0, np.array([x, 0.0, y, 0.0, z, 0.0])),
                (
                    t0 + datetime.timedelta(seconds=100),
                    np.array([x, 0.0, y, 0.0, z, 0.0]),
                ),
            ],
        )
        red_stub = _StationaryTargetController(target_id=1, point=p_target)
        return rng, terrain, t0, dt, p_launcher, track, red_stub

    def _make_simulator(
        self, terrain, blue_controller, red_stub, blue_tracker, t0, dt, rng
    ) -> Simulator:
        return Simulator(
            pcl_detector=PclDetector(),
            pet_detector=PetDetector(terrain_model=terrain),
            blue_controller=blue_controller,
            red_controller=red_stub,
            blue_tracker=blue_tracker,
            red_tracker=DummyTracker(),
            start_time=t0,
            time_step=dt,
            min_time_per_step=datetime.timedelta(seconds=0),
            termination_criterion=TimeCriterion(t0 + datetime.timedelta(seconds=1000)),
            rng=rng,
            listener=InMemoryLogger(),
            terrain_model=terrain,
            damage_model=UniformDamageModel(1.0, rng),  # every real hit would kill
        )

    def test_direct_fire_consumes_ammo_without_a_kill(self):
        rng, terrain, t0, dt, p_launcher, track, red_stub = self._build_scenario()

        effector = DirectFireEffector(
            id=5,
            point=p_launcher,
            combat_range=100_000,
            n_attacks_left=3,
            name="",
            terrain=terrain,
            cadence=float("inf"),
        )
        launcher = StaticGbadController(
            target_id=4,
            sidc=SIDC.BLUE_AIR_DEFENCE,
            rcs=1.5,
            effector=effector,
            assigned_track_id="9",
        )
        blue_group = ControllerGroup([launcher])
        simulator = self._make_simulator(
            terrain, blue_group, red_stub, _FixedTrackTracker(track), t0, dt, rng
        )
        recorder = _EventRecorder()
        simulator.register_event_listener(recorder)

        # Tick 0:
        # The targets come into existance only after the attacks in tick 0.
        # Therefore, there are no shots and no ammo consumption.
        self.assertTrue(simulator.advance())
        self.assertEqual(effector.n_attacks_left, 3)

        # Tick 1:
        # Ground truth now exists. The track is far enough from it
        # that no real target associates, but the effector still fires at
        # it (ammo consumed) - and no kill results.
        self.assertTrue(simulator.advance())
        self.assertEqual(effector.n_attacks_left, 2)
        self.assertEqual(
            len([e for e in recorder.events if isinstance(e, DirectShot)]), 1
        )
        self.assertEqual(
            len([e for e in recorder.events if isinstance(e, KillEvent)]), 0
        )

    def test_indirect_fire_consumes_ammo_and_still_launches(self):
        rng, terrain, t0, dt, p_launcher, track, red_stub = self._build_scenario()

        projectile_effector = DirectFireEffector(
            id=99,
            point=p_launcher,
            combat_range=5_000,
            n_attacks_left=1,
            name="",
            terrain=terrain,
            cadence=float("inf"),
        )
        effector = IndirectFireEffector(
            id=5,
            point=p_launcher,
            combat_range=100_000,
            n_attacks_left=3,
            name="",
            projectile=projectile_effector,
            projectile_speed=300.0,
            projectile_max_dist=50_000.0,
            projectile_sidc=SIDC.BLUE_MISSILE,
            projectile_rcs=ConstantRcsModel(rcs=0.1),
            cadence=float("inf"),
        )
        launcher = StaticGbadController(
            target_id=4,
            sidc=SIDC.BLUE_AIR_DEFENCE,
            rcs=1.5,
            effector=effector,
            assigned_track_id="9",
        )
        blue_group = ControllerGroup([launcher])
        simulator = self._make_simulator(
            terrain, blue_group, red_stub, _FixedTrackTracker(track), t0, dt, rng
        )
        recorder = _EventRecorder()
        simulator.register_event_listener(recorder)

        self.assertTrue(simulator.advance())
        self.assertEqual(effector.n_attacks_left, 3)

        # The tracking error doesn't stop the launch (a real missile leaves
        # the tube either way). It just means the projectile is aimed at a
        # track with no real target actually behind it.
        self.assertTrue(simulator.advance())
        self.assertEqual(effector.n_attacks_left, 2)
        self.assertEqual(
            len([e for e in recorder.events if isinstance(e, IndirectShot)]), 1
        )
        spawned = [c for c in blue_group._controllers if c is not launcher]
        self.assertEqual(len(spawned), 1)
        self.assertEqual(spawned[0].child.assigned_track_id, "9")


if __name__ == "__main__":
    unittest.main()
