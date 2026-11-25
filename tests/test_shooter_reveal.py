import datetime
import unittest

from theia.config import UNKNOWN_RCS
from theia.effectors import DirectFireEffector
from theia.simulation.controllers.intel_controller import IntelController
from theia.simulation.controllers.shooter_reveal_controller import (
    ShooterRevealController,
)
from theia.terrain import AbstractTerrainModel
from theia.types import (
    AbstractEventListener,
    ConstantRcsModel,
    Controller,
    DirectShot,
    Entity,
    Event,
    IdProvider,
    IndirectShot,
    IntelSensor,
    KillEvent,
    Party,
    Point,
    SimulationContext,
    SituationalPicture,
    Target,
    TargetInfos,
    TextEvent,
    TheiaException,
    Velocity,
)

t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
dt = datetime.timedelta(seconds=1)
VALID_TIME = datetime.timedelta(seconds=10)

P_SHOOTER = Point(lat=47.0, lon=8.0, alt=500.0)
P_TARGET = Point(lat=47.0, lon=8.01, alt=1000.0)


class FlatTerrain(AbstractTerrainModel):
    step_m: float = 30.0

    def elevationAt(self, lat, lon):
        return 0.0

    def has_line_of_sight(self, p1: Point, p2: Point) -> bool:
        return True


def get_effector(effector_id: int = 0) -> DirectFireEffector:
    return DirectFireEffector(
        id=effector_id,
        point=P_SHOOTER,
        combat_range=100_000,
        n_attacks_left=10,
        name="",
        terrain=FlatTerrain(),
        cadence=float("inf"),
    )


def get_target(party: Party) -> Target:
    return Target(
        id=1,
        is_stationary=False,
        info=TargetInfos.FIXED_WING.with_party(party),
        point=P_TARGET,
        cross_section_model=ConstantRcsModel(rcs=1.0),
        velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
    )


def get_direct_shot(
    target_party: Party,
    time: datetime.datetime = t0,
    shooter_id: int = 0,
) -> DirectShot:
    return DirectShot(
        id=-1,
        time=time,
        shooter=get_effector(shooter_id),
        target=get_target(target_party),
    )


def get_indirect_shot(
    target_party: Party, time: datetime.datetime = t0
) -> IndirectShot:
    return IndirectShot(
        id=-1,
        time=time,
        shooter=get_effector(0),
        projectile=get_effector(100),
        target=get_target(target_party),
        track_id="track",
    )


def get_picture(time: datetime.datetime = t0) -> SituationalPicture:
    return SituationalPicture(
        time=time,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


def get_controller(
    is_blue: bool,
    id_provider: IdProvider | None = None,
) -> ShooterRevealController:
    controller = ShooterRevealController(is_blue=is_blue, valid_time=VALID_TIME)
    controller.bind(
        SimulationContext(
            id_provider=id_provider if id_provider is not None else IdProvider(),
            start_time=t0,
        )
    )
    return controller


def own_party(is_blue: bool) -> Party:
    return Party.from_is_blue(is_blue)


def enemy_party(is_blue: bool) -> Party:
    return Party.from_is_blue(not is_blue)


def n_children(controller: ShooterRevealController) -> int:
    return len(controller._controllers)


class RecordingController(Controller):
    def model_post_init(self, context):
        super().model_post_init(context)
        self._events = []

    def update(self, situational_picture, dt):
        pass

    def on_event(self, event):
        self._events.append(event)


class ShooterRevealControllerTest(unittest.TestCase):
    def test_starts_without_children(self):
        self.assertEqual(n_children(get_controller(True)), 0)

    def test_enemy_shot_requires_binding(self):
        # Test that receiving an enemy shot without
        # binding a simulation throws an error.
        # Receiving a friendly shot does not raise an error since no new IDs are
        # because there is no intel gain.
        controller = ShooterRevealController(is_blue=True, valid_time=VALID_TIME)
        controller.on_event(get_direct_shot(Party.RED))
        with self.assertRaises(TheiaException):
            controller.on_event(get_direct_shot(Party.BLUE))

    def test_spawned_children_are_bound(self):
        controller = get_controller(True)
        controller.on_event(get_direct_shot(Party.BLUE))
        self.assertIs(
            controller._controllers[0].simulation_context,
            controller.simulation_context,
        )

    def test_enemy_shot_spawns_child(self):
        for is_blue in (True, False):
            with self.subTest(is_blue=is_blue):
                controller = get_controller(is_blue)
                controller.on_event(get_direct_shot(own_party(is_blue)))
                self.assertEqual(n_children(controller), 1)
                self.assertIsInstance(controller._controllers[0], IntelController)

    def test_own_shot_spawns_no_child(self):
        for is_blue in (True, False):
            with self.subTest(is_blue=is_blue):
                controller = get_controller(is_blue)
                controller.on_event(get_direct_shot(enemy_party(is_blue)))
                self.assertEqual(n_children(controller), 0)

    def test_unattributable_target_spawns_no_child(self):
        for is_blue in (True, False):
            for party in (Party.UNKNOWN, Party.NEUTRAL):
                with self.subTest(is_blue=is_blue, party=party):
                    controller = get_controller(is_blue)
                    controller.on_event(get_direct_shot(party))
                    self.assertEqual(n_children(controller), 0)

    def test_non_shot_events_spawn_no_child(self):
        controller = get_controller(True)
        controller.on_event(Event(id=1, time=t0))
        controller.on_event(TextEvent(id=2, time=t0, text="hello"))
        controller.on_event(KillEvent(id=3, time=t0, target_id=1))
        self.assertEqual(n_children(controller), 0)

    def test_indirect_shot_behaves_like_direct_shot(self):
        for is_blue in (True, False):
            with self.subTest(is_blue=is_blue):
                controller = get_controller(is_blue)
                controller.on_event(get_indirect_shot(enemy_party(is_blue)))
                self.assertEqual(n_children(controller), 0)
                controller.on_event(get_indirect_shot(own_party(is_blue)))
                self.assertEqual(n_children(controller), 1)

                controller.update(get_picture(t0 + VALID_TIME), dt)
                self.assertEqual(n_children(controller), 0)

    def test_repeated_shots_spawn_one_child_each(self):
        controller = get_controller(True)
        controller.on_event(get_direct_shot(Party.BLUE, shooter_id=0))
        controller.on_event(get_direct_shot(Party.BLUE, shooter_id=0))
        self.assertEqual(n_children(controller), 2)
        self.assertIsNot(controller._controllers[0], controller._controllers[1])

    def test_child_expires_exactly_at_valid_time(self):
        for is_blue in (True, False):
            with self.subTest(is_blue=is_blue):
                controller = get_controller(is_blue)
                controller.on_event(get_direct_shot(own_party(is_blue)))

                controller.update(get_picture(t0), dt)
                self.assertEqual(n_children(controller), 1)
                controller.update(
                    get_picture(t0 + VALID_TIME - datetime.timedelta(microseconds=1)),
                    dt,
                )
                self.assertEqual(n_children(controller), 1)
                controller.update(get_picture(t0 + VALID_TIME), dt)
                self.assertEqual(n_children(controller), 0)
                self.assertEqual(controller._expiries, [])

    def test_children_expire_independently(self):
        controller = get_controller(True)
        t_second = t0 + datetime.timedelta(seconds=3)
        controller.on_event(get_direct_shot(Party.BLUE, time=t0))
        controller.on_event(get_direct_shot(Party.BLUE, time=t_second))
        second_child = controller._controllers[1]

        controller.update(get_picture(t0 + VALID_TIME), dt)
        self.assertEqual(controller._controllers, [second_child])
        self.assertIs(controller._controllers[0], second_child)

        controller.update(get_picture(t_second + VALID_TIME), dt)
        self.assertEqual(n_children(controller), 0)

    def test_outputs_are_one_intel_sensor_per_child(self):
        controller = get_controller(True)
        controller.on_event(get_direct_shot(Party.BLUE))
        controller.on_event(get_indirect_shot(Party.BLUE))
        controller.update(get_picture(t0), dt)

        self.assertEqual(n_children(controller), 2)

        self.assertEqual(controller.monostatic_sensors, [])
        self.assertEqual(controller.pcl_sensors, [])
        self.assertEqual(controller.targets, [])
        self.assertEqual(controller.pet_receivers, [])
        self.assertEqual(controller.visual_sensors, [])
        self.assertEqual(
            controller.intel_sensors,
            [c.sensor for c in controller._controllers],
        )
        self.assertEqual(controller.firing_effectors, [])
        self.assertEqual(controller.geojson, [])

    def test_intel_sensors_vanish_after_expiry(self):
        controller = get_controller(True)
        controller.on_event(get_direct_shot(Party.BLUE))
        controller.update(get_picture(t0), dt)
        self.assertEqual(len(controller.intel_sensors), 1)
        controller.update(get_picture(t0 + VALID_TIME), dt)
        self.assertEqual(controller.intel_sensors, [])

    def test_sensor_reveals_shooter_position(self):
        controller = get_controller(True)
        controller.on_event(get_direct_shot(Party.BLUE))
        sensor = controller._controllers[0].sensor
        self.assertIsInstance(sensor, IntelSensor)
        # The span is unlimited: valid at the shot time, decades later and
        # far in the past; the child's lifetime is limited by valid_time.
        for time in (
            t0,
            datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
            datetime.datetime(1900, 1, 1, tzinfo=datetime.UTC),
        ):
            with self.subTest(time=time):
                target = sensor.detect(time)
                self.assertIsNotNone(target)
                self.assertAlmostEqual(target.point.lat, P_SHOOTER.lat)
                self.assertAlmostEqual(target.point.lon, P_SHOOTER.lon)
                self.assertAlmostEqual(target.point.alt, P_SHOOTER.alt)
                self.assertEqual(target.info, TargetInfos.UNKNOWN)
                self.assertEqual(target.cross_section_model.rcs, UNKNOWN_RCS)

    def test_ids_come_from_id_provider(self):
        id_provider = IdProvider()
        id_provider.register_entity(Entity.TARGET, 41)
        id_provider.register_entity(Entity.SENSOR, 6)
        controller = get_controller(True, id_provider)
        controller.on_event(get_direct_shot(Party.BLUE))
        controller.on_event(get_direct_shot(Party.BLUE))

        sensors = [c.sensor for c in controller._controllers]
        self.assertEqual([s.id for s in sensors], [7, 8])
        self.assertEqual([s.trajectory.target_id for s in sensors], [42, 43])
        self.assertEqual(id_provider.increment(Entity.SENSOR), 9)
        self.assertEqual(id_provider.increment(Entity.TARGET), 44)

    def test_events_are_forwarded_to_children(self):
        controller = get_controller(True)
        recorder = RecordingController()
        controller.add_controller(recorder)

        shot = get_direct_shot(Party.BLUE)
        own_shot = get_direct_shot(Party.RED)
        text = TextEvent(id=2, time=t0, text="hello")
        for event in (shot, own_shot, text):
            controller.on_event(event)

        self.assertEqual(recorder._events, [shot, own_shot, text])

    def test_spawned_child_events_reach_listeners(self):
        controller = get_controller(True)
        received = []

        class Listener(AbstractEventListener):
            def on_event(self, event):
                received.append(event)

        controller.register_event_listener(Listener())
        controller.on_event(get_direct_shot(Party.BLUE))
        event = TextEvent(id=5, time=t0, text="from child")
        controller._controllers[0]._broadcast_event(event)

        self.assertEqual(received, [event])


if __name__ == "__main__":
    unittest.main()
