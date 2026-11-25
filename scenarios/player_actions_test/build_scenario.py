"""
Build a game scenario for testing the player / action interface in the frontend.

BLUE (not player-controlled):
    5 long-range (LORAD) indirect-fire GBAD units on the Swiss plateau, each
    with a medium range radar next to it, plus 2 long-range surveillance
    radars on mountain tops, and 5 critical infrastructure facilities.
    A short-range (SHORAD) direct-fire GBAD on the Uetliberg publishes its
    effector range as GeoJSON.

Every radar of both sides comes with a precomputed coverage at 5000 MASL,
published as GeoJSON.

RED (player-controlled):
    2 surveillance radars (switchable on/off by the player) and 2 launch
    locations. Each location holds 3 ballistic missiles and 5 one-way drones;
    location A additionally holds 2 cruise missiles. A depot further back
    holds reserve ballistic missiles and drones, which the player can move to
    the launch sites. Ballistic missiles can also be moved between the two
    launch locations.

A camp site keeps stock of exactly one ammunition type, so each launch
location (and the depot) consists of one camp site per ammunition type. The camp sites of a
location are offset by a few hundred metres so that their map markers can be
told apart.

Both sides have a shooter-reveal intel source: a shooter of the other side
that fires becomes known at its position for a limited time. RED thereby learns
where the BLUE GBADs that intercept its rounds are. (Launches from camp sites
emit no shot event yet, so BLUE learns nothing about RED's launch sites.)
RED also has lasting intel on all BLUE critical infrastructure: its positions
are known from the start for the whole game.

Besides the scenario file, the BLUE and RED ORBATs are written next to it as
``<name>_orbat_blue.json`` and ``<name>_orbat_red.json``, for inspection in the
scenario editor. They are taken from the built scenario, so their IDs match
it.

Usage (from the repository root):

    python scenarios/player_actions_test/build_scenario.py \
        scenarios/player_actions_test/player_actions_test.json

    python scripts/simulate_scenario.py \
        scenarios/player_actions_test/player_actions_test.json \
        output/player_actions_test.json --interactive
"""

import argparse
import datetime
import json
import pathlib

from theia.config import DEFAULT_TERRAIN
from theia.simulation.scenario_import import (
    BallisticMissileTemplate,
    CampSite,
    CriticalInfrastructureFactory,
    CruiseMissileTemplate,
    DirectFireEffectorFactory,
    IndirectFireEffectorFactory,
    KnownTargetFactory,
    MonostaticCoverageCalcSettings,
    MonostaticCoverageCalculation,
    MonostaticSensorFactory,
    OneWayDroneTemplate,
    OrderOfBattle,
    ShooterRevealFactory,
    SimulationResults,
    StaticGbadFactory,
    TerrainFactory,
    TransportTime,
    build_scenario,
)
from theia.coverage import calculate_coverage
from theia.radar_equation import calculate_maximum_monostatic_range
from theia.types import (
    ConstantRcsModel,
    Entity,
    GeoJSONFeature,
    IdProvider,
    MonostaticSensor,
    Party,
    Point,
    TargetInfos,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "data/default_configurations"
SURVEILLANCE_RADAR_CONFIG = CONFIG_DIR / "surveillance_radar.json"
MEDIUM_RANGE_RADAR_CONFIG = CONFIG_DIR / "medium_range_radar.json"

SCENARIO_NAME = "Player actions test"
SEED = 42

# Height of effectors above ground [m]
EFFECTOR_ALTITUDE_OFFSET = 5.0
# Offset between the camp sites of one launch location [deg latitude]
CAMP_SITE_OFFSET_LAT = 0.003  # approx. 330 m

# LORAD GBAD parameters (Patriot-like, rough public figures)
LORAD_RCS = 50.0
LORAD_COMBAT_RANGE = 100_000.0
LORAD_N_ATTACKS = 16
LORAD_CADENCE = 0.5
LORAD_MAX_IN_FLIGHT = 1  # one interceptor at a time; the next one after it is resolved
INTERCEPTOR_SPEED = 1_200.0
INTERCEPTOR_MAX_DIST = 120_000.0
INTERCEPTOR_KILL_RANGE = 2_000.0
INTERCEPTOR_RCS = 0.1

BLUE_GBADS = {
    "LORAD Zürich": (47.45, 8.56),
    "LORAD Basel": (47.50, 7.75),
    "LORAD Bern": (46.95, 7.45),
    "LORAD Luzern": (47.05, 8.30),
    "LORAD St. Gallen": (47.42, 9.37),
}
# SHORAD GBAD: direct fire, publishes its effector range as GeoJSON
SHORAD_NAME = "SHORAD Uetliberg"
SHORAD_POSITION = (47.3497, 8.4912)
SHORAD_RCS = 10.0
SHORAD_COMBAT_RANGE = 15_000.0
SHORAD_N_ATTACKS = 10
SHORAD_CADENCE = 1.0
SHORAD_GEOJSON_RANGE_ALTITUDES = {1000, 3000}
"""Altitudes of the published effector ranges [m above sea level]"""

# Offset of a GBAD's medium range radar from its launcher [deg longitude]
GBAD_RADAR_OFFSET_LON = 0.004  # approx. 300 m
BLUE_RADAR_RCS = 10.0
BLUE_SURVEILLANCE_RADARS = {
    "Surveillance radar Pilatus": (46.979, 8.252),
    "Surveillance radar Chasseral": (47.133, 7.054),
}

# Name -> (lat, lon, tag)
BLUE_CRITICAL_INFRASTRUCTURE = {
    "Kernkraftwerk Gösgen": (47.366, 7.967, "nuclear_power_plant"),
    "Kernkraftwerk Leibstadt": (47.601, 8.183, "nuclear_power_plant"),
    "Unterwerk Laufenburg": (47.555, 8.065, "substation"),
    "Flughafen Zürich": (47.458, 8.555, "airport"),
    "Bundeshaus Bern": (46.947, 7.444, "government"),
}

RED_RADAR_RCS = 20.0
RED_RADARS = {
    "Radar Feldberg": (47.874, 8.004),
    "Radar Schwäbische Alb": (48.32, 9.20),
}

# Precomputed radar coverages (defaults of the scenario editor)
COVERAGE_SETTINGS = MonostaticCoverageCalcSettings(
    targetAlt=5000.0,
    targetRcs=1.0,
    probabilityThreshold=0.8,
    latRes=0.01,
    lonRes=0.025,
)

# How long a revealed shooter stays known after its shot [s]
SHOOTER_REVEAL_SECONDS_VALID = 300.0

CAMP_SITE_RCS = 10.0
LAUNCH_LOCATIONS = {
    "A": (48.45, 8.40),
    "B": (48.40, 9.95),
}
DEPOT_LOCATION = (48.70, 9.10)  # near Stuttgart
# Reserve stock of the depot: round name -> number of rounds
DEPOT_STOCK = {"BM": 6, "Drones": 10}
# Transport routes (one-way; a return trip needs its own entry):
# (source site name, target site name, seconds until leave, seconds underway)
TRANSPORT_ROUTES = [
    ("Depot (BM)", "Launch site A (BM)", 600.0, 3_600.0),
    ("Depot (BM)", "Launch site B (BM)", 600.0, 4_200.0),
    ("Depot (Drones)", "Launch site A (Drones)", 300.0, 3_600.0),
    ("Depot (Drones)", "Launch site B (Drones)", 300.0, 4_200.0),
    ("Launch site A (BM)", "Launch site B (BM)", 900.0, 6_900.0),
    ("Launch site B (BM)", "Launch site A (BM)", 900.0, 6_900.0),
]
BALLISTIC_MISSILE = BallisticMissileTemplate(rcs=0.5)
ONEWAY_DRONE = OneWayDroneTemplate(rcs=0.1, speed=50.0)
CRUISE_MISSILE = CruiseMissileTemplate(rcs=0.1, speed=250.0)


def ground_point(terrain, lat: float, lon: float, offset: float = 0.0) -> Point:
    return Point(lat=lat, lon=lon, alt=terrain.elevationAt(lat, lon) + offset)


def empty_orbat(**kwargs) -> OrderOfBattle:
    defaults = {
        "monostatic_sensors": [],
        "pcl_sensors": [],
        "gbads": [],
        "simulationResults": {
            "monostaticCoverages": [],
            "pclMinDetectableRcsGrids": [],
        },
        "oneway_drones": [],
        "ballistic_missiles": [],
    }
    return OrderOfBattle(**(defaults | kwargs))


def load_radar_template(path: pathlib.Path) -> dict:
    with open(path) as file:
        return json.load(file)["sensor"]


def build_radar(
    template: dict,
    sensor_id: int,
    target_id: int,
    rcs: float,
    name: str,
    point: Point,
) -> MonostaticSensorFactory:
    sensor = MonostaticSensor.model_validate(template)
    sensor.id = sensor_id
    sensor.transmitter.id = sensor_id
    sensor.receiver.id = sensor_id
    sensor.transmitter.point = point
    sensor.receiver.point = point
    return MonostaticSensorFactory(
        target_id=target_id, rcs=rcs, sensor=sensor, name=name
    )


def build_blue_orbat(terrain) -> OrderOfBattle:
    medium_range = load_radar_template(MEDIUM_RANGE_RADAR_CONFIG)
    surveillance = load_radar_template(SURVEILLANCE_RADAR_CONFIG)

    target_id = 1
    sensor_id = 1
    radars = []
    gbads = []
    for i, (name, (lat, lon)) in enumerate(BLUE_GBADS.items(), start=1):
        radars.append(
            build_radar(
                medium_range,
                sensor_id,
                target_id,
                BLUE_RADAR_RCS,
                f"{name} radar",
                ground_point(terrain, lat, lon + GBAD_RADAR_OFFSET_LON),
            )
        )
        sensor_id += 1
        target_id += 1
        point = ground_point(terrain, lat, lon, EFFECTOR_ALTITUDE_OFFSET)
        gbads.append(
            StaticGbadFactory(
                target_id=target_id,
                rcs=LORAD_RCS,
                name=name,
                gbad=IndirectFireEffectorFactory(
                    id=i,
                    name=name,
                    point=point,
                    combat_range=LORAD_COMBAT_RANGE,
                    n_attacks_left=LORAD_N_ATTACKS,
                    cadence=LORAD_CADENCE,
                    max_in_flight=LORAD_MAX_IN_FLIGHT,
                    projectile=DirectFireEffectorFactory(
                        # Template only: each shot mints a fresh ID.
                        id=0,
                        name=f"{name} interceptor",
                        point=point,
                        combat_range=INTERCEPTOR_KILL_RANGE,
                        n_attacks_left=1,
                        cadence=1.0,
                    ),
                    projectile_speed=INTERCEPTOR_SPEED,
                    projectile_max_dist=INTERCEPTOR_MAX_DIST,
                    projectile_rcs=ConstantRcsModel(rcs=INTERCEPTOR_RCS),
                ),
            )
        )
        target_id += 1

    for name, (lat, lon) in BLUE_SURVEILLANCE_RADARS.items():
        radars.append(
            build_radar(
                surveillance,
                sensor_id,
                target_id,
                BLUE_RADAR_RCS,
                name,
                ground_point(terrain, lat, lon),
            )
        )
        sensor_id += 1
        target_id += 1

    infrastructure = []
    for name, (lat, lon, tag) in BLUE_CRITICAL_INFRASTRUCTURE.items():
        infrastructure.append(
            CriticalInfrastructureFactory(
                target_id=target_id,
                name=name,
                point=ground_point(terrain, lat, lon),
                tags=frozenset({tag}),
            )
        )
        target_id += 1

    gbads.append(
        StaticGbadFactory(
            target_id=target_id,
            rcs=SHORAD_RCS,
            name=SHORAD_NAME,
            gbad=DirectFireEffectorFactory(
                id=len(gbads) + 1,
                name=SHORAD_NAME,
                point=ground_point(terrain, *SHORAD_POSITION, EFFECTOR_ALTITUDE_OFFSET),
                combat_range=SHORAD_COMBAT_RANGE,
                n_attacks_left=SHORAD_N_ATTACKS,
                cadence=SHORAD_CADENCE,
            ),
            geojson_range_altitudes=SHORAD_GEOJSON_RANGE_ALTITUDES,
        )
    )
    target_id += 1

    return empty_orbat(
        monostatic_sensors=radars,
        gbads=gbads,
        critical_infrastructure=infrastructure,
        unused_id_sensor=sensor_id,
        unused_id_receiver=sensor_id,
        unused_id_transmitter=sensor_id,
        intel=[ShooterRevealFactory(seconds_valid=SHOOTER_REVEAL_SECONDS_VALID)],
        unused_id_effector=len(gbads) + 1,
        unused_target_id=target_id,
        is_user_controlled=False,
    )


def build_red_orbat(terrain) -> OrderOfBattle:
    surveillance = load_radar_template(SURVEILLANCE_RADAR_CONFIG)

    target_id = 1
    radars = []
    for i, (name, (lat, lon)) in enumerate(RED_RADARS.items(), start=1):
        radars.append(
            build_radar(
                surveillance,
                i,
                target_id,
                RED_RADAR_RCS,
                name,
                ground_point(terrain, lat, lon),
            )
        )
        target_id += 1

    stock = {
        "A": [
            (BALLISTIC_MISSILE, 3, 300.0),
            (ONEWAY_DRONE, 5, 30.0),
            (CRUISE_MISSILE, 2, 120.0),
        ],
        "B": [(BALLISTIC_MISSILE, 3, 300.0), (ONEWAY_DRONE, 5, 30.0)],
    }
    round_names = {
        "ballistic_missile": "BM",
        "oneway_drone": "Drones",
        "cruise_missile": "CM",
    }
    camp_sites = []
    site_id = 1
    for location, (lat, lon) in LAUNCH_LOCATIONS.items():
        for k, (ammo, n_ammo, seconds_between_launches) in enumerate(stock[location]):
            site_lat = lat + k * CAMP_SITE_OFFSET_LAT
            camp_sites.append(
                CampSite(
                    id=site_id,
                    name=f"Launch site {location} ({round_names[ammo.type]})",
                    point=ground_point(terrain, site_lat, lon),
                    target_id=target_id,
                    rcs=CAMP_SITE_RCS,
                    n_ammo=n_ammo,
                    ammo=ammo,
                    can_launch=True,
                    seconds_between_launches=seconds_between_launches,
                )
            )
            site_id += 1
            target_id += 1

    ammo_by_name = {"BM": BALLISTIC_MISSILE, "Drones": ONEWAY_DRONE}
    lat, lon = DEPOT_LOCATION
    for k, (round_name, n_ammo) in enumerate(DEPOT_STOCK.items()):
        camp_sites.append(
            CampSite(
                id=site_id,
                name=f"Depot ({round_name})",
                point=ground_point(terrain, lat + k * CAMP_SITE_OFFSET_LAT, lon),
                target_id=target_id,
                rcs=CAMP_SITE_RCS,
                n_ammo=n_ammo,
                ammo=ammo_by_name[round_name],
                can_launch=False,
                seconds_between_launches=0.0,
            )
        )
        site_id += 1
        target_id += 1

    site_ids = {site.name: site.id for site in camp_sites}
    transport_times = [
        TransportTime(
            source_site_id=site_ids[source],
            target_site_id=site_ids[target],
            seconds_until_leave=seconds_until_leave,
            seconds_underway=seconds_underway,
        )
        for source, target, seconds_until_leave, seconds_underway in TRANSPORT_ROUTES
    ]

    n_radars = len(radars)
    return empty_orbat(
        monostatic_sensors=radars,
        camp_sites=camp_sites,
        transport_times=transport_times,
        intel=[ShooterRevealFactory(seconds_valid=SHOOTER_REVEAL_SECONDS_VALID)]
        + known_blue_infrastructure(terrain),
        unused_id_sensor=n_radars + 1,
        unused_id_receiver=n_radars + 1,
        unused_id_transmitter=n_radars + 1,
        unused_id_effector=0,
        unused_target_id=target_id,
        unused_id_camp_site=site_id,
        is_user_controlled=True,
    )


def known_blue_infrastructure(terrain) -> list[KnownTargetFactory]:
    """RED's lasting intel on the BLUE critical infrastructure (exact positions)."""
    info = TargetInfos.CRITICAL_INFRASTRUCTURE.with_party(Party.BLUE)
    return [
        KnownTargetFactory(
            name=name, point=ground_point(terrain, lat, lon), target_info=info
        )
        for name, (lat, lon, _) in BLUE_CRITICAL_INFRASTRUCTURE.items()
    ]


def with_free_ids(orbat: OrderOfBattle) -> None:
    """
    Raise the ``unused_*`` counters above every ID in use.

    ``OrderOfBattle.reindex`` (called on RED by ``build_scenario``) renumbers
    the entities but not the counters, so ``OrderOfBattle.from_file`` would
    reject the ORBAT as inconsistent.
    """
    id_provider = IdProvider()
    orbat.update_id_provider(id_provider)
    free = id_provider._free_ids
    orbat.unused_id_sensor = max(orbat.unused_id_sensor, free[Entity.SENSOR])
    orbat.unused_id_receiver = max(orbat.unused_id_receiver, free[Entity.RECEIVER])
    orbat.unused_id_transmitter = max(
        orbat.unused_id_transmitter, free[Entity.TRANSMITTER]
    )
    orbat.unused_id_effector = max(orbat.unused_id_effector, free[Entity.EFFECTOR])
    orbat.unused_target_id = max(orbat.unused_target_id, free[Entity.TARGET])
    orbat.unused_id_camp_site = max(orbat.unused_id_camp_site, free[Entity.CAMP_SITE])


def add_coverages(orbat: OrderOfBattle, terrain):
    """
    Precompute the coverage of every radar, like the scenario editor does.

    Must run after `build_scenario`, which may reassign the sensor IDs.
    """
    settings = COVERAGE_SETTINGS
    now = datetime.datetime.now(datetime.UTC)
    calculations = []
    for factory in orbat.monostatic_sensors:
        radar = factory.sensor
        max_dist = calculate_maximum_monostatic_range(
            radar=radar,
            target_rcs=settings.targetRcs,
            probability_threshold=settings.probabilityThreshold,
        )
        polygons = calculate_coverage(
            terrain,
            radar.receiver.antenna_point,
            max_dist,
            settings.targetAlt,
            settings.latRes,
            settings.lonRes,
        )
        calculation = MonostaticCoverageCalculation(
            sensorId=radar.id,
            settings=settings,
            coverage=[GeoJSONFeature.from_shapely(p) for p in polygons],
        )
        calculations.append((calculation, now))
        print(f"Coverage of {factory.name or radar.id}: {max_dist / 1000:.0f} km")
    orbat.simulationResults = SimulationResults(
        monostaticCoverages=calculations, pclMinDetectableRcsGrids=[]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("output_file", type=pathlib.Path)
    args = parser.parse_args()

    terrain_factory = TerrainFactory(terrain_name=DEFAULT_TERRAIN)
    terrain = terrain_factory.to_terrain()

    scenario = build_scenario(
        SCENARIO_NAME,
        terrain_factory,
        SEED,
        build_blue_orbat(terrain),
        build_red_orbat(terrain),
    )
    add_coverages(scenario.blue_orbat, terrain)
    add_coverages(scenario.red_orbat, terrain)

    with open(args.output_file, "w") as file:
        file.write(scenario.model_dump_json(indent=4))

    for side, orbat in (("blue", scenario.blue_orbat), ("red", scenario.red_orbat)):
        with_free_ids(orbat)
        orbat_file = args.output_file.with_name(
            f"{args.output_file.stem}_orbat_{side}.json"
        )
        with open(orbat_file, "w") as file:
            file.write(orbat.model_dump_json(indent=4))
