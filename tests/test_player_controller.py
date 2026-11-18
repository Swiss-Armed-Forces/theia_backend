import datetime
import threading
import unittest

from theia.simulation.actions import (
    LaunchAction,
    MoveStockAction,
    ToggleMonostaticRadarAction,
)
from theia.simulation.controllers.intel_controller import IntelController
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.player_controller import (
    PlayerController,
    ShipmentState,
)
from theia.simulation.scenario_import import (
    CampSite,
    CriticalInfrastructureFactory,
    MonostaticSensorFactory,
    OneWayDroneTemplate,
    OrderOfBattle,
    TransportTime,
)
from theia.terrain import DummyTerrain
from theia.test_data import build_flores_monostatic_radar
from theia.types import (
    AbstractEventListener,
    ConstantRcsModel,
    Entity,
    Event,
    IdProvider,
    IntelSensor,
    KillEvent,
    Party,
    Point,
    ShipmentLostEvent,
    SimulationContext,
    SituationalPicture,
    TargetInfos,
    TextEvent,
    TheiaException,
    Trajectory,
)

T0 = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.UTC)
DT = datetime.timedelta(seconds=1)

LAUNCH_SITE_ID = 1
DEPOT_ID = 2
LAUNCH_SITE_TARGET_ID = 10
DEPOT_TARGET_ID = 11
RADAR_SENSOR_ID = 5
RADAR_TARGET_ID = 12
INFRA_TARGET_ID = 13
SECONDS_BETWEEN_LAUNCHES = 30.0
SECONDS_UNTIL_LEAVE = 5.0
SECONDS_UNDERWAY = 10.0

DESTINATION = Point(lat=47.0, lon=8.1, alt=1.0)


def build_orbat(n_launch_site: int = 2, n_depot: int = 10) -> OrderOfBattle:
    drone = OneWayDroneTemplate(rcs=0.1, speed=50.0)
    return OrderOfBattle(
        monostatic_sensors=[
            MonostaticSensorFactory(
                target_id=RADAR_TARGET_ID,
                rcs=10.0,
                sensor=build_flores_monostatic_radar(
                    Point(lat=46.9, lon=8.0, alt=1.0), RADAR_SENSOR_ID, 0, 0
                ),
            )
        ],
        pcl_sensors=[],
        gbads=[],
        simulationResults={"monostaticCoverages": [], "pclMinDetectableRcsGrids": []},
        oneway_drones=[],
        ballistic_missiles=[],
        critical_infrastructure=[
            CriticalInfrastructureFactory(
                target_id=INFRA_TARGET_ID,
                name="power plant",
                point=Point(lat=46.95, lon=8.0, alt=1.0),
            )
        ],
        camp_sites=[
            CampSite(
                id=LAUNCH_SITE_ID,
                name="launch site",
                point=Point(lat=47.0, lon=8.0, alt=1.0),
                target_id=LAUNCH_SITE_TARGET_ID,
                rcs=10.0,
                n_ammo=n_launch_site,
                ammo=drone,
                can_launch=True,
                seconds_between_launches=SECONDS_BETWEEN_LAUNCHES,
            ),
            CampSite(
                id=DEPOT_ID,
                name="depot",
                point=Point(lat=47.05, lon=8.0, alt=1.0),
                target_id=DEPOT_TARGET_ID,
                rcs=10.0,
                n_ammo=n_depot,
                ammo=drone,
                can_launch=False,
                seconds_between_launches=0.0,
            ),
        ],
        transport_times=[
            TransportTime(
                source_site_id=DEPOT_ID,
                target_site_id=LAUNCH_SITE_ID,
                seconds_until_leave=SECONDS_UNTIL_LEAVE,
                seconds_underway=SECONDS_UNDERWAY,
            ),
            TransportTime(
                source_site_id=LAUNCH_SITE_ID,
                target_site_id=DEPOT_ID,
                seconds_until_leave=SECONDS_UNTIL_LEAVE,
                seconds_underway=SECONDS_UNDERWAY,
            ),
        ],
        unused_id_sensor=100,
        unused_id_receiver=100,
        unused_id_transmitter=100,
        unused_id_effector=100,
        unused_target_id=100,
        is_user_controlled=True,
    )


class _Recorder(AbstractEventListener):
    def __init__(self):
        self.events: list[Event] = []

    def on_event(self, event: Event):
        self.events.append(event)

    def texts(self, marker: str) -> list[str]:
        return [
            e.text for e in self.events if isinstance(e, TextEvent) and marker in e.text
        ]


class PlayerControllerTestBase(unittest.TestCase):
    def setUp(self):
        self.build()

    def build(self, **kwargs):
        orbat = build_orbat(**kwargs)
        self.id_provider = IdProvider()
        orbat.update_id_provider(self.id_provider)
        self.controller = orbat.to_controller(DummyTerrain(has_los=True), True)
        self.assertIsInstance(self.controller, PlayerController)
        self.context = SimulationContext(id_provider=self.id_provider)
        self.controller.bind(self.context)
        self.recorder = _Recorder()
        self.controller.register_event_listener(self.recorder)
        self.time = T0
        self.step()

    def step(self, n: int = 1):
        for _ in range(n):
            self.controller.update(
                SituationalPicture(
                    time=self.time,
                    friendly_pet_receivers=[],
                    friendly_radars=[],
                    friendly_targets=[],
                    enemy_targets=[],
                ),
                DT,
            )
            self.time += DT

    def act(self, action):
        self.assertTrue(self.controller.assign_action(action))
        self.step()

    def site(self, site_id: int):
        [site] = [s for s in self.controller.get_status().sites if s.id == site_id]
        return site

    def target_ids(self) -> set[int]:
        return {t.id for t in self.controller.targets}


class LaunchTest(PlayerControllerTestBase):
    def launch(self, site_id: int = LAUNCH_SITE_ID):
        self.act(LaunchAction(site_id=site_id, destination=DESTINATION))

    def test_launch_reduces_stock_and_adds_round(self):
        n_targets = len(self.target_ids())
        next_target_id = self.id_provider._free_ids[Entity.TARGET]
        self.launch()
        self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 1)
        self.assertEqual(len(self.target_ids()), n_targets + 1)
        self.assertEqual(len(self.recorder.texts("executed")), 1)
        # New IDs come from the shared ID provider.
        self.assertIn(next_target_id, self.target_ids())
        self.assertEqual(self.id_provider._free_ids[Entity.TARGET], next_target_id + 1)

    def test_launch_leaves_order_of_battle_untouched(self):
        orbat = build_orbat()
        self.controller = orbat.to_controller(DummyTerrain(has_los=True), True)
        self.controller.bind(self.context)
        self.step()
        self.launch()
        self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 1)
        [launch_site] = [s for s in orbat.camp_sites if s.id == LAUNCH_SITE_ID]
        self.assertEqual(launch_site.n_ammo, 2)

    def test_round_attacks_destination(self):
        target_id = self.id_provider._free_ids[Entity.TARGET]
        effector_id = self.id_provider._free_ids[Entity.EFFECTOR]
        self.launch()
        [round_controller] = [
            c
            for c in self.controller._controllers
            if getattr(c, "target_id", None) == target_id
        ]
        drone = round_controller.child
        self.assertEqual(drone.assigned_goal, DESTINATION)
        self.assertEqual(drone.effector.n_attacks_left, 1)
        self.assertEqual(drone.effector.cadence, float("inf"))
        self.assertEqual(drone.effector.id, effector_id)

    def test_launch_respects_cooldown(self):
        self.launch()
        self.launch()
        self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 1)
        [rejection] = self.recorder.texts("rejected")
        self.assertIn("reloading", rejection)
        # The rejected launch is not held back for later.
        self.step(int(SECONDS_BETWEEN_LAUNCHES) + 1)
        self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 1)
        self.launch()
        self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 0)

    def test_seconds_until_ready(self):
        self.launch()
        self.assertGreater(self.site(LAUNCH_SITE_ID).seconds_until_ready, 0.0)
        self.step(int(SECONDS_BETWEEN_LAUNCHES))
        self.assertEqual(self.site(LAUNCH_SITE_ID).seconds_until_ready, 0.0)

    def test_launch_from_depot_is_rejected(self):
        self.launch(DEPOT_ID)
        self.assertEqual(self.site(DEPOT_ID).n_ammo, 10)
        [rejection] = self.recorder.texts("rejected")
        self.assertIn("cannot launch", rejection)

    def test_launch_with_empty_stock_is_rejected(self):
        self.build(n_launch_site=0)
        self.launch()
        [rejection] = self.recorder.texts("rejected")
        self.assertIn("no stock", rejection)

    def test_launch_of_stock_in_transit_is_rejected(self):
        self.act(
            MoveStockAction(
                source_site_id=LAUNCH_SITE_ID, target_site_id=DEPOT_ID, count=2
            )
        )
        self.launch()
        [rejection] = self.recorder.texts("rejected")
        self.assertIn("no stock", rejection)

    def test_unknown_site_is_rejected(self):
        self.launch(site_id=99)
        [rejection] = self.recorder.texts("rejected")
        self.assertIn("unknown camp site", rejection)

    def test_every_action_is_logged(self):
        self.launch()
        self.launch(DEPOT_ID)
        self.assertEqual(len(self.recorder.texts("received")), 2)
        self.assertEqual(len(self.recorder.texts("executed")), 1)
        self.assertEqual(len(self.recorder.texts("rejected")), 1)
        self.assertEqual(
            [e.time for e in self.recorder.events],
            [T0 + DT, T0 + DT, T0 + 2 * DT, T0 + 2 * DT],
        )


class MoveStockTest(PlayerControllerTestBase):
    def move(self, count: int = 3):
        self.act(
            MoveStockAction(
                source_site_id=DEPOT_ID, target_site_id=LAUNCH_SITE_ID, count=count
            )
        )

    def test_move_timing(self):
        t_order = self.time
        self.move()
        status = self.controller.get_status()
        [shipment] = status.shipments
        self.assertEqual(shipment.state, ShipmentState.BREAKING_CAMP)
        self.assertEqual(
            shipment.arrival_time,
            t_order
            + datetime.timedelta(seconds=SECONDS_UNTIL_LEAVE + SECONDS_UNDERWAY),
        )
        # Taken out of the depot's stock at once ...
        self.assertEqual(self.site(DEPOT_ID).n_ammo, 7)
        # ... but the depot stays visible.
        self.assertIn(DEPOT_TARGET_ID, self.target_ids())

        n_targets = len(self.target_ids())
        while self.time <= t_order + datetime.timedelta(seconds=SECONDS_UNTIL_LEAVE):
            status = self.controller.get_status()
            [shipment] = status.shipments
            self.assertEqual(shipment.state, ShipmentState.BREAKING_CAMP)
            self.step()
        [shipment] = self.controller.get_status().shipments
        self.assertEqual(shipment.state, ShipmentState.UNDERWAY)
        # The shipment is invisible.
        self.assertEqual(len(self.target_ids()), n_targets)
        self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 2)

        while self.time < shipment.arrival_time:
            [shipment] = self.controller.get_status().shipments
            self.assertEqual(shipment.state, ShipmentState.UNDERWAY)
            self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 2)
            self.step()
        # The update at the arrival time delivers.
        self.step()
        self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 5)
        self.assertEqual(self.controller.get_status().shipments, [])

    def test_move_more_than_stock_is_rejected(self):
        self.move(count=11)
        self.assertEqual(self.site(DEPOT_ID).n_ammo, 10)
        self.assertEqual(len(self.recorder.texts("rejected")), 1)

    def test_move_without_route_is_rejected(self):
        self.act(
            MoveStockAction(source_site_id=DEPOT_ID, target_site_id=DEPOT_ID, count=1)
        )
        [rejection] = self.recorder.texts("rejected")
        self.assertIn("no route", rejection)

    def test_shipment_to_destroyed_site_is_lost(self):
        self.move()
        self.controller.on_event(
            KillEvent(id=1, time=self.time, target_id=LAUNCH_SITE_TARGET_ID)
        )
        self.step(int(SECONDS_UNTIL_LEAVE + SECONDS_UNDERWAY) + 1)
        self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 0)
        self.assertEqual(self.controller.get_status().shipments, [])
        [event] = [e for e in self.recorder.events if isinstance(e, ShipmentLostEvent)]
        self.assertEqual(event.party, Party.BLUE)
        self.assertEqual(event.shipment_id, 0)
        self.assertEqual(event.source_site_id, DEPOT_ID)
        self.assertEqual(event.source_site_name, "depot")
        self.assertEqual(event.target_site_id, LAUNCH_SITE_ID)
        self.assertEqual(event.target_site_name, "launch site")
        self.assertEqual(event.destroyed_site_id, LAUNCH_SITE_ID)
        self.assertEqual(event.count, 3)
        self.assertEqual(event.destroyed_site_name, "launch site")
        self.assertIn(
            "shipment #0 lost: 3 rounds, camp site launch site (#1) is destroyed",
            str(event),
        )


class CampSiteKillTest(PlayerControllerTestBase):
    def test_kill_destroys_stock_and_shipments_breaking_camp(self):
        self.act(
            MoveStockAction(
                source_site_id=DEPOT_ID, target_site_id=LAUNCH_SITE_ID, count=4
            )
        )
        self.controller.on_event(
            KillEvent(id=1, time=self.time, target_id=DEPOT_TARGET_ID)
        )
        self.step()
        depot = self.site(DEPOT_ID)
        self.assertFalse(depot.is_alive)
        self.assertEqual(depot.n_ammo, 0)
        self.assertEqual(self.controller.get_status().shipments, [])
        self.assertNotIn(DEPOT_TARGET_ID, self.target_ids())
        [text] = self.recorder.texts("destroyed")
        self.assertIn("6 rounds in stock", text)
        self.assertIn("shipment #0 (4 rounds)", text)
        [event] = [e for e in self.recorder.events if isinstance(e, ShipmentLostEvent)]
        self.assertEqual(event.party, Party.BLUE)
        self.assertEqual(event.shipment_id, 0)
        self.assertEqual(event.source_site_id, DEPOT_ID)
        self.assertEqual(event.source_site_name, "depot")
        self.assertEqual(event.target_site_id, LAUNCH_SITE_ID)
        self.assertEqual(event.target_site_name, "launch site")
        self.assertEqual(event.destroyed_site_id, DEPOT_ID)
        self.assertEqual(event.count, 4)
        self.assertEqual(event.destroyed_site_name, "depot")
        self.assertIn(
            "shipment #0 lost: 4 rounds, camp site depot (#2) is destroyed",
            str(event),
        )

    def test_shipment_underway_survives_kill_of_source(self):
        self.act(
            MoveStockAction(
                source_site_id=DEPOT_ID, target_site_id=LAUNCH_SITE_ID, count=4
            )
        )
        self.step(int(SECONDS_UNTIL_LEAVE) + 1)
        self.controller.on_event(
            KillEvent(id=1, time=self.time, target_id=DEPOT_TARGET_ID)
        )
        self.step(int(SECONDS_UNDERWAY) + 1)
        self.assertEqual(self.site(LAUNCH_SITE_ID).n_ammo, 6)
        self.assertFalse(
            any(isinstance(e, ShipmentLostEvent) for e in self.recorder.events)
        )

    def test_launch_from_destroyed_site_is_rejected(self):
        self.controller.on_event(
            KillEvent(id=1, time=self.time, target_id=LAUNCH_SITE_TARGET_ID)
        )
        self.act(LaunchAction(site_id=LAUNCH_SITE_ID, destination=DESTINATION))
        [rejection] = self.recorder.texts("rejected")
        self.assertIn("destroyed", rejection)


class RadarToggleTest(PlayerControllerTestBase):
    def radar(self) -> MonostaticRadarController:
        [radar] = self.controller._radars.values()
        return radar.child.child

    def radar_target(self):
        [target] = [t for t in self.controller.targets if t.id == RADAR_TARGET_ID]
        return target

    def test_radar_is_on_by_default(self):
        self.assertEqual(
            [s.id for s in self.controller.monostatic_sensors], [RADAR_SENSOR_ID]
        )
        self.assertIsNotNone(self.radar_target().transmitter)

    def test_switch_off_and_on(self):
        self.act(
            ToggleMonostaticRadarAction(sensor_id=RADAR_SENSOR_ID, new_is_on=False)
        )
        self.assertEqual(list(self.controller.monostatic_sensors), [])
        self.assertIsNone(self.radar_target().transmitter)
        # The wrapped controller is untouched.
        radar = self.radar()
        self.assertIsInstance(radar, MonostaticRadarController)
        self.assertEqual(len(radar.monostatic_sensors), 1)
        [target] = radar.targets
        self.assertIsNotNone(target.transmitter)

        self.act(ToggleMonostaticRadarAction(sensor_id=RADAR_SENSOR_ID, new_is_on=True))
        self.assertEqual(len(list(self.controller.monostatic_sensors)), 1)
        self.assertIsNotNone(self.radar_target().transmitter)

    def test_unknown_radar_is_rejected(self):
        self.act(ToggleMonostaticRadarAction(sensor_id=99, new_is_on=False))
        [rejection] = self.recorder.texts("rejected")
        self.assertIn("unknown monostatic radar", rejection)


class OutputTest(PlayerControllerTestBase):
    def test_camp_sites_are_targets(self):
        self.assertIn(LAUNCH_SITE_TARGET_ID, self.target_ids())
        self.assertIn(DEPOT_TARGET_ID, self.target_ids())

    def test_camp_sites_are_part_of_the_tree(self):
        [tree] = self.controller._controllers
        group = tree.child
        for site in self.controller._sites.values():
            self.assertIn(site, group._controllers)

    def test_intel_sensors_are_forwarded(self):
        sensor = IntelSensor(
            id=50,
            trajectory=Trajectory(
                target_id=51,
                target_info=TargetInfos.UNKNOWN,
                times=[T0, T0 + DT],
                lats=[47.0] * 2,
                lons=[8.0] * 2,
                alts=[1.0] * 2,
                vxs=[0.0] * 2,
                vys=[0.0] * 2,
                vzs=[0.0] * 2,
                cross_section_model=ConstantRcsModel(rcs=1.0),
            ),
        )
        self.controller.add_controller(IntelController(sensor=sensor))
        self.step()
        self.assertEqual(self.controller.intel_sensors, [sensor])


class BindingTest(PlayerControllerTestBase):
    def test_child_tree_is_bound(self):
        [tree] = self.controller._controllers
        self.assertIs(tree.simulation_context, self.context)
        [radar] = self.controller._radars.values()
        self.assertIs(radar.child.child.simulation_context, self.context)

    def test_launched_round_is_bound(self):
        self.act(LaunchAction(site_id=LAUNCH_SITE_ID, destination=DESTINATION))
        # Launched rounds are added to the PlayerController itself.
        [tree, round_controller] = self.controller._controllers
        self.assertIs(round_controller.simulation_context, self.context)

    def test_launch_requires_binding(self):
        controller = build_orbat().to_controller(DummyTerrain(has_los=True), True)
        controller.assign_action(
            LaunchAction(site_id=LAUNCH_SITE_ID, destination=DESTINATION)
        )
        with self.assertRaises(TheiaException):
            controller.update(
                SituationalPicture(
                    time=T0,
                    friendly_pet_receivers=[],
                    friendly_radars=[],
                    friendly_targets=[],
                    enemy_targets=[],
                ),
                DT,
            )


class ConcurrencyTest(PlayerControllerTestBase):
    def test_concurrent_assign_action(self):
        n_threads = 8
        n_per_thread = 50

        def worker():
            for _ in range(n_per_thread):
                self.controller.assign_action(
                    ToggleMonostaticRadarAction(
                        sensor_id=RADAR_SENSOR_ID, new_is_on=False
                    )
                )

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        # Run the simulation concurrently with the producers.
        while any(t.is_alive() for t in threads):
            self.step()
        for t in threads:
            t.join()
        self.step()
        self.assertEqual(len(self.recorder.texts("received")), n_threads * n_per_thread)
        self.assertEqual(len(self.recorder.texts("executed")), n_threads * n_per_thread)


class LegacyOrbatTest(unittest.TestCase):
    def test_user_controlled_without_camp_sites(self):
        orbat = build_orbat()
        orbat.camp_sites = []
        orbat.transport_times = []
        controller = orbat.to_controller(DummyTerrain(has_los=True), True)
        self.assertIsInstance(controller, PlayerController)

    def test_not_user_controlled_no_player_controller(self):
        orbat = build_orbat()
        orbat.is_user_controlled = False
        controller = orbat.to_controller(DummyTerrain(has_los=True), True)
        self.assertNotIsInstance(controller, PlayerController)
        self.assertFalse(
            controller.assign_action(
                ToggleMonostaticRadarAction(sensor_id=RADAR_SENSOR_ID, new_is_on=False)
            )
        )


if __name__ == "__main__":
    unittest.main()
