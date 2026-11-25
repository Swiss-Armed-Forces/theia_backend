import datetime
import types
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from tests.test_action_controller import (
    DEPOT_ID,
    LAUNCH_SITE_ID,
    RADAR_SENSOR_ID,
    RADAR_TARGET_ID,
    SECONDS_UNDERWAY,
    SECONDS_UNTIL_LEAVE,
    build_orbat,
)
from tests.test_scenario_import import get_simulation_results
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.damage_model import UniformDamageModel
from theia.simulation.scenario_import import (
    CampSite,
    OneWayDroneTemplate,
    OrderOfBattle,
)
from theia.simulation.server import EventMessage, create_app
from theia.simulation.simulator import Simulator, TimeCriterion
from theia.simulation.theia_logging import SituationalPictureBuffer
from theia.simulation.trackers.tracking import DummyTracker
from theia.terrain import DummyTerrain
from theia.test_data import build_flores_monostatic_radar
from theia.types import (
    ConstantRcsModel,
    IdProvider,
    KillEvent,
    Party,
    Point,
    SituationalPicture,
    Target,
    TargetInfos,
    Track,
    TrackInitEvent,
    Velocity,
)

T0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)


def get_client() -> TestClient:
    buffer = SituationalPictureBuffer()
    target = Target(
        id=7,
        is_stationary=True,
        name="GBAD",
        info=TargetInfos.GBAD.with_party(Party.BLUE),
        point=Point(lat=47.0, lon=8.0, alt=500.0),
        cross_section_model=ConstantRcsModel(rcs=1.0),
        velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
    )
    buffer.on_snapshot(
        types.SimpleNamespace(time=T0, blue_targets=[target], red_targets=[])
    )
    track = Track(
        id="42",
        target_info=TargetInfos.FIXED_WING.with_party(Party.RED),
        states=[
            (T0, [4.3e6, 0.0, 6.0e5, 0.0, 4.6e6, 0.0]),
            (T0 + datetime.timedelta(seconds=1), [4.3e6, 0.0, 6.0e5, 0.0, 4.6e6, 0.0]),
        ],
    )
    buffer.on_situational_picture(
        SituationalPicture(
            time=T0,
            friendly_pet_receivers=[],
            friendly_radars=[],
            friendly_targets=[],
            enemy_targets=[track],
        ),
        is_blue=True,
    )
    return TestClient(create_app(buffer, director=None))


class ServerApiTest(unittest.TestCase):
    def test_ground_truth_contains_info(self):
        response = get_client().post("/ground_truth/BLUE", json=[T0.isoformat()])
        self.assertEqual(response.status_code, 200)
        [gt] = response.json()
        self.assertEqual(gt["sidc"], "10031000001301000000")
        self.assertEqual(gt["info"]["category"], "GBAD")
        self.assertEqual(gt["info"]["party"], "BLUE")

    def test_situational_picture_contains_info(self):
        response = get_client().post("/situational_picture/BLUE", json=[T0.isoformat()])
        self.assertEqual(response.status_code, 200)
        [track] = response.json()["enemy_tracks"]
        self.assertEqual(track["sidc"], "10260100001101000000")
        self.assertEqual(track["info"]["party"], "RED")

    def test_openapi_schema_exposes_target_info(self):
        schemas = create_app(None, None).openapi()["components"]["schemas"]
        self.assertIn("TargetInfo", schemas)
        self.assertIn("TargetCategory", schemas)


class MonostaticCoverageTest(unittest.TestCase):
    def test_coverage_starts_at_antenna(self):
        """The coverage must be calculated from the receiver's antenna position."""
        radar = build_flores_monostatic_radar(
            Point(lat=47.0, lon=8.0, alt=500.0), 0, 1, 2
        )
        self.assertGreater(radar.receiver.antenna_height, 0.0)
        director = types.SimpleNamespace(
            _simulator=types.SimpleNamespace(_terrain_model=None)
        )
        client = TestClient(create_app(SituationalPictureBuffer(), director))
        with (
            patch(
                "theia.simulation.server.calculate_maximum_monostatic_range",
                return_value=10_000.0,
            ),
            patch(
                "theia.simulation.server.calculate_coverage", return_value=[]
            ) as calculate_coverage,
        ):
            response = client.post(
                "/calculate_monostatic_coverage",
                params=dict(
                    target_alt=1000.0,
                    target_rcs=1.0,
                    probability_threshold=0.5,
                    lat_res=0.01,
                    lon_res=0.01,
                ),
                json=radar.model_dump(mode="json"),
            )
        self.assertEqual(response.status_code, 200, response.text)
        point = calculate_coverage.call_args.args[1]
        self.assertEqual(point, radar.receiver.antenna_point)


RED_CAMP_SITE_ID = 3
RED_CAMP_SITE_TARGET_ID = 20


def get_game_client() -> tuple[TestClient, Simulator]:
    """
    Client for a running game: BLUE is user-controlled, RED is not.

    BLUE's radar has a precomputed coverage.
    """
    terrain = DummyTerrain(has_los=True)
    blue_orbat = build_orbat()
    blue_orbat.simulationResults = get_simulation_results(RADAR_SENSOR_ID)
    red_orbat = OrderOfBattle(
        monostatic_sensors=[],
        pcl_sensors=[],
        gbads=[],
        simulationResults={"monostaticCoverages": [], "pclMinDetectableRcsGrids": []},
        oneway_drones=[],
        ballistic_missiles=[],
        # A camp site on a side that is not user-controlled: simulated as a
        # target, but not reported by /camp_sites.
        camp_sites=[
            CampSite(
                id=RED_CAMP_SITE_ID,
                name="red launch site",
                point=Point(lat=46.5, lon=8.0, alt=1.0),
                target_id=RED_CAMP_SITE_TARGET_ID,
                rcs=10.0,
                n_ammo=5,
                ammo=OneWayDroneTemplate(rcs=0.1, speed=50.0),
                can_launch=True,
                seconds_between_launches=30.0,
            )
        ],
        unused_id_sensor=100,
        unused_id_receiver=100,
        unused_id_transmitter=100,
        unused_id_effector=100,
        unused_target_id=100,
    )
    id_provider = IdProvider()
    blue_orbat.update_id_provider(id_provider)
    red_orbat.update_id_provider(id_provider)
    rng = np.random.Generator(np.random.PCG64(seed=1))
    buffer = SituationalPictureBuffer()
    simulator = Simulator(
        pcl_detector=PclDetector(),
        pet_detector=PetDetector(terrain_model=terrain),
        blue_controller=blue_orbat.to_controller(terrain, True),
        red_controller=red_orbat.to_controller(terrain, False),
        blue_tracker=DummyTracker(),
        red_tracker=DummyTracker(),
        start_time=T0,
        time_step=datetime.timedelta(seconds=1),
        min_time_per_step=datetime.timedelta(seconds=0),
        termination_criterion=TimeCriterion(T0 + datetime.timedelta(seconds=1000)),
        rng=rng,
        listener=buffer,
        terrain_model=terrain,
        damage_model=UniformDamageModel(1.0, rng),
        simulate_clutter=False,
        id_provider=id_provider,
    )
    return TestClient(create_app(buffer, director=None)), simulator


LAUNCH = {
    "type": "launch",
    "site_id": LAUNCH_SITE_ID,
    "destination": {"lat": 47.0, "lon": 8.1, "alt": 1.0},
}


class GameApiTest(unittest.TestCase):
    def setUp(self):
        self.client, self.simulator = get_game_client()

    def assign(self, which: str, action: dict) -> bool:
        response = self.client.post(f"/assign_action/{which}", json=action)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def messages(self) -> list[str]:
        return [e["msg"] for e in self.client.get("/events").json()]

    def sites(self, which: str = "BLUE") -> dict[int, dict]:
        status = self.client.get(f"/camp_sites/{which}").json()
        return {s["id"]: s for s in status["sites"]}

    def test_is_user_controlled(self):
        self.assertIs(self.client.get("/is_user_controlled/BLUE").json(), True)
        self.assertIs(self.client.get("/is_user_controlled/RED").json(), False)

    def test_unknown_team_is_rejected(self):
        response = self.client.get("/is_user_controlled/GREEN")
        self.assertEqual(response.status_code, 422)

    def test_launch(self):
        self.assertIs(self.assign("BLUE", LAUNCH), True)
        self.simulator.advance()
        site = self.sites()[LAUNCH_SITE_ID]
        self.assertEqual(site["n_ammo"], 1)
        self.assertGreater(site["seconds_until_ready"], 0.0)
        self.assertTrue(
            any(
                m.startswith("[ACTION BLUE]") and "executed" in m
                for m in self.messages()
            )
        )

    def test_side_that_is_not_user_controlled_refuses_actions(self):
        self.assertIs(self.assign("RED", LAUNCH), False)
        self.simulator.advance()
        self.assertEqual(self.sites()[LAUNCH_SITE_ID]["n_ammo"], 2)
        self.assertFalse(any("[ACTION" in m for m in self.messages()))

    def test_toggle_radar(self):
        action = {
            "type": "toggle_monostatic_radar",
            "sensor_id": RADAR_SENSOR_ID,
            "new_is_on": False,
        }
        self.assertIs(self.assign("BLUE", action), True)
        self.simulator.advance()
        self.assertTrue(
            any(
                m.startswith("[ACTION BLUE]") and "executed" in m
                for m in self.messages()
            )
        )
        self.assertEqual(self.simulator._blue_monostatic_radars, [])

    def test_impossible_action_is_accepted_and_rejected_as_event(self):
        action = {
            "type": "toggle_monostatic_radar",
            "sensor_id": 99,
            "new_is_on": False,
        }
        self.assertIs(self.assign("BLUE", action), True)
        self.simulator.advance()
        self.assertTrue(any("rejected (UNKNOWN_RADAR)" in m for m in self.messages()))

    def test_order_fields_set_by_the_client_are_overwritten(self):
        self.assign("BLUE", {**LAUNCH, "id": 12345, "party": "RED"})
        self.simulator.advance()
        messages = self.messages()
        self.assertFalse(any("#12345" in m for m in messages))
        self.assertFalse(any(m.startswith("[ACTION RED]") for m in messages))
        self.assertTrue(any(m.startswith("[ACTION BLUE]") for m in messages))

    def test_malformed_action(self):
        for body in (
            {"type": "launch", "site_id": LAUNCH_SITE_ID},
            {"type": "idle"},
            {"site_id": LAUNCH_SITE_ID, "destination": LAUNCH["destination"]},
            {
                "type": "move_stock",
                "source_site_id": DEPOT_ID,
                "target_site_id": LAUNCH_SITE_ID,
                "count": 0,
            },
        ):
            with self.subTest(body=body):
                response = self.client.post("/assign_action/BLUE", json=body)
                self.assertEqual(response.status_code, 422)

    def test_camp_sites(self):
        sites = self.sites()
        self.assertEqual(set(sites), {LAUNCH_SITE_ID, DEPOT_ID})
        self.assertEqual(sites[DEPOT_ID]["n_ammo"], 10)
        self.assertTrue(sites[DEPOT_ID]["is_alive"])
        self.assertTrue(sites[LAUNCH_SITE_ID]["can_launch"])
        self.assertEqual(sites[LAUNCH_SITE_ID]["seconds_until_ready"], 0.0)

    def test_camp_sites_shows_shipments(self):
        action = {
            "type": "move_stock",
            "source_site_id": DEPOT_ID,
            "target_site_id": LAUNCH_SITE_ID,
            "count": 3,
        }
        self.assertIs(self.assign("BLUE", action), True)
        self.simulator.advance()
        status = self.client.get("/camp_sites/BLUE").json()
        sites = {s["id"]: s for s in status["sites"]}
        self.assertEqual(sites[DEPOT_ID]["n_ammo"], 7)
        [shipment] = status["shipments"]
        self.assertEqual(shipment["source_site_id"], DEPOT_ID)
        self.assertEqual(shipment["target_site_id"], LAUNCH_SITE_ID)
        self.assertEqual(shipment["count"], 3)
        self.assertEqual(shipment["state"], "BREAKING_CAMP")
        self.assertIn("departure_time", shipment)
        self.assertIn("arrival_time", shipment)

    def test_camp_sites_time_before_first_step_is_start_time(self):
        status = self.client.get("/camp_sites/BLUE").json()
        self.assertEqual(status["time"], self.client.get("/time").json())

    def test_camp_sites_of_side_that_is_not_user_controlled(self):
        # The camp site exists in the simulation, but nobody commands it.
        self.simulator.advance()
        self.assertIn(
            RED_CAMP_SITE_TARGET_ID, {t.id for t in self.simulator._red_targets}
        )
        response = self.client.get("/camp_sites/RED")
        self.assertEqual(response.status_code, 200, response.text)
        status = response.json()
        self.assertEqual(status["sites"], [])
        self.assertEqual(status["shipments"], [])
        self.assertIsNotNone(status["time"])

    def test_transport_times(self):
        routes = self.client.get("/transport_times/BLUE").json()
        self.assertEqual(
            {(r["source_site_id"], r["target_site_id"]) for r in routes},
            {(DEPOT_ID, LAUNCH_SITE_ID), (LAUNCH_SITE_ID, DEPOT_ID)},
        )
        for route in routes:
            self.assertEqual(route["seconds_until_leave"], SECONDS_UNTIL_LEAVE)
            self.assertEqual(route["seconds_underway"], SECONDS_UNDERWAY)
        self.assertEqual(self.client.get("/transport_times/RED").json(), [])

    def test_openapi_schema_exposes_actions(self):
        schemas = create_app(None, None).openapi()["components"]["schemas"]
        for name in (
            "LaunchAction",
            "MoveStockAction",
            "ToggleMonostaticRadarAction",
            "CampSitesStatus",
            "TransportTime",
        ):
            self.assertIn(name, schemas)

    def test_action_event_messages_belong_to_the_actor(self):
        self.assertIs(self.assign("BLUE", LAUNCH), True)
        self.simulator.advance()
        # A second launch from the same site is rejected: it is reloading.
        self.assertIs(self.assign("BLUE", LAUNCH), True)
        self.simulator.advance()
        events = self.client.get("/events").json()
        action_events = [e for e in events if e["msg"].startswith("[ACTION")]
        # Two actions, one executed, one rejected.
        self.assertEqual(len(action_events), 4)
        self.assertTrue(any("executed" in e["msg"] for e in action_events))
        self.assertTrue(any("rejected" in e["msg"] for e in action_events))
        for event in action_events:
            self.assertEqual(event["party"], "BLUE", event["msg"])
        logged_ids = [e["id"] for e in events if e["id"] >= 0]
        self.assertEqual(len(logged_ids), len(set(logged_ids)))

    def test_kill_event_messages_belongs_to_the_victim(self):
        event = KillEvent(id=3, time=T0, target_id=7, party=Party.RED)
        message = EventMessage.from_event(event)
        self.assertEqual(message.id, 3)
        self.assertEqual(message.party, Party.RED)
        self.assertEqual(message.model_dump(mode="json")["party"], "RED")

    def test_event_without_party(self):
        event = TrackInitEvent(id=4, time=T0, target_id=7, track_id=1)
        self.assertIsNone(EventMessage.from_event(event).party)



class GeoJsonApiTest(unittest.TestCase):
    def setUp(self):
        self.client, self.simulator = get_game_client()

    def geojson(self, which: str) -> list[dict]:
        response = self.client.get(f"/geojson/{which}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_radar_coverage_is_served_as_annotation(self):
        self.simulator.advance()

        [annotation] = self.geojson("BLUE")
        self.assertEqual(annotation["id"], f"sensor-range-{RADAR_SENSOR_ID}-1000-1.5")
        self.assertEqual(annotation["kind"], "SENSOR_RANGE")
        self.assertEqual(annotation["target_ids"], [RADAR_TARGET_ID])
        self.assertEqual(
            annotation["hover_text"],
            f"Monostatic #{RADAR_SENSOR_ID}, 1000 MASL, RCS=1.5m^2",
        )
        self.assertEqual(annotation["feature"]["type"], "Feature")
        self.assertEqual(annotation["feature"]["geometry"]["type"], "MultiPolygon")
        self.assertEqual(self.geojson("RED"), [])

    def test_geojson_is_live(self):
        self.simulator.advance()
        self.assertEqual(len(self.geojson("BLUE")), 1)

        action = {
            "type": "toggle_monostatic_radar",
            "sensor_id": RADAR_SENSOR_ID,
            "new_is_on": False,
        }
        self.client.post("/assign_action/BLUE", json=action)
        self.simulator.advance()

        self.assertEqual(self.geojson("BLUE"), [])

    def test_openapi_schema_exposes_geojson_annotation(self):
        schemas = create_app(None, None).openapi()["components"]["schemas"]
        self.assertIn("GeoJSONAnnotation", schemas)
        self.assertEqual(schemas["GeoJSONKind"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
