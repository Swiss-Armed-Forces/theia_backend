import datetime
import json
import logging
import os
import tempfile
import unittest
from typing import ClassVar

import numpy as np
import pydantic
import shapely

from theia.distance import line_of_sight_distance
from theia.effectors import DirectFireEffector, IndirectFireEffector
from theia.simulation.controllers.critical_infrastructure_controller import (
    CriticalInfrastructureController,
)
from theia.simulation.controllers.fixed_path_kamikaze_drone import FixedPathOneWayDrone
from theia.simulation.controllers.known_target_controller import (
    KnownTargetController,
)
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.shooter_reveal_controller import (
    ShooterRevealController,
)
from theia.simulation.path_generators import BallisticGenerator
from theia.simulation.scenario_import import (
    BallisticMissileFactory,
    BallisticMissileTemplate,
    CampSite,
    CriticalInfrastructureFactory,
    CruiseMissileFactory,
    CruiseMissileTemplate,
    DirectFireEffectorFactory,
    FixedPathOneWayDroneFactory,
    IndirectFireEffectorFactory,
    KnownTargetFactory,
    MonostaticCoverageCalcSettings,
    MonostaticCoverageCalculation,
    MonostaticSensorFactory,
    NeverTermination,
    OneWayDroneTemplate,
    OrderOfBattle,
    ScenarioFactory,
    ShooterRevealFactory,
    SimulationResults,
    StaticGbadFactory,
    TerrainFactory,
    TimeTermination,
    Trajectory,
    TransportTime,
    _AmmoTemplateBase,
    build_scenario,
)
from theia.simulation.simulator import NeverCriterion, TimeCriterion
from theia.terrain import ConstantSphereTerrainModel, SrtmTerrainModel
from theia.test_data import get_uetliberg_radar
from theia.types import (
    ConstantRcsModel,
    DirectShot,
    Entity,
    GeoJSONFeature,
    GeoJSONKind,
    IdProvider,
    Party,
    Point,
    SimulationContext,
    SituationalPicture,
    Target,
    TargetCategory,
    TargetInfos,
    Velocity,
)

srtm = SrtmTerrainModel()

p = Point(lat=47.37348, lon=8.53707, alt=1000.0)


def get_direct_factory() -> DirectFireEffectorFactory:
    return DirectFireEffectorFactory(
        id=1,
        name="direct",
        point=p,
        combat_range=5_000,
        n_attacks_left=1,
        cadence=1.0,
    )


def get_indirect_factory() -> IndirectFireEffectorFactory:
    return IndirectFireEffectorFactory(
        id=2,
        name="indirect",
        point=p,
        combat_range=100_000,
        n_attacks_left=1,
        cadence=1.0,
        projectile=DirectFireEffectorFactory(
            id=3,
            name="projectile",
            point=p,
            combat_range=5_000,
            n_attacks_left=1,
            cadence=1.0,
        ),
        projectile_speed=300.0,
        projectile_max_dist=50_000.0,
        projectile_rcs=ConstantRcsModel(rcs=0.1),
    )


def get_order_of_battle(
    gbads: list[StaticGbadFactory],
    critical_infrastructure: list[CriticalInfrastructureFactory] | None = None,
    ballistic_missiles: list[BallisticMissileFactory] | None = None,
    oneway_drones: list[FixedPathOneWayDroneFactory] | None = None,
    cruise_missiles: list[CruiseMissileFactory] | None = None,
) -> OrderOfBattle:
    return OrderOfBattle(
        monostatic_sensors=[],
        pcl_sensors=[],
        gbads=gbads,
        simulationResults={"monostaticCoverages": [], "pclMinDetectableRcsGrids": []},
        oneway_drones=oneway_drones or [],
        ballistic_missiles=ballistic_missiles or [],
        critical_infrastructure=critical_infrastructure or [],
        cruise_missiles=cruise_missiles or [],
        unused_id_sensor=0,
        unused_id_receiver=0,
        unused_id_transmitter=0,
        unused_id_effector=0,
        unused_target_id=0,
    )


def get_ballistic_missile_factory(
    target_id: int,
    effector_id: int,
    rcs: float = 1.0,
    alpha: float = 45.0,
) -> BallisticMissileFactory:
    return BallisticMissileFactory(
        p_start=Point(lat=47.0, lon=8.0, alt=0.0),
        p_stop=Point(lat=47.1, lon=8.1, alt=0.0),
        t_start=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        terrain=TerrainFactory(terrain_name="SRTM"),
        target_id=target_id,
        effector_id=effector_id,
        rcs=rcs,
        alpha=alpha,
    )


def get_cruise_missile_factory(
    target_id: int,
    effector_id: int,
    **kwargs,
) -> CruiseMissileFactory:
    return CruiseMissileFactory(
        p_start=Point(lat=47.0, lon=8.0, alt=0.0),
        p_stop=Point(lat=47.1, lon=8.1, alt=0.0),
        t_start=datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.UTC),
        terrain=TerrainFactory(terrain_name="SRTM"),
        target_id=target_id,
        effector_id=effector_id,
        rcs=0.1,
        speed=250.0,
        cruise_magl=80.0,
        **kwargs,
    )


def get_oneway_drone_factory(target_id: int) -> FixedPathOneWayDroneFactory:
    t0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    t1 = datetime.datetime(2024, 1, 1, 0, 0, 1, tzinfo=datetime.UTC)
    return FixedPathOneWayDroneFactory(
        effector=get_direct_factory(),
        trajectory=Trajectory(
            target_id=target_id,
            target_info=TargetInfos.for_category(TargetCategory.DRONE_CLASS_I),
            times=[t0, t1],
            lats=[p.lat, p.lat],
            lons=[p.lon, p.lon],
            alts=[p.alt, p.alt],
            cross_section_model=ConstantRcsModel(rcs=1.0),
        ),
        assigned_goal=p,
    )


class IndirectFireEffectorFactoryTest(unittest.TestCase):
    def test_to_effector_blue(self):
        effector = get_indirect_factory().to_effector(srtm, is_blue=True)
        self.assertIsInstance(effector, IndirectFireEffector)
        self.assertEqual(effector.id, 2)
        self.assertEqual(effector.projectile.id, 3)
        self.assertIsInstance(effector.projectile, DirectFireEffector)
        self.assertEqual(effector.projectile_speed, 300.0)
        self.assertEqual(effector.projectile_max_dist, 50_000.0)
        self.assertEqual(
            effector.projectile_info,
            TargetInfos.INTERCEPTOR.with_party(Party.BLUE),
        )

    def test_to_effector_red(self):
        effector = get_indirect_factory().to_effector(srtm, is_blue=False)
        self.assertEqual(
            effector.projectile_info,
            TargetInfos.INTERCEPTOR.with_party(Party.RED),
        )


class StaticGbadFactoryTest(unittest.TestCase):
    def test_to_controller_direct(self):
        factory = StaticGbadFactory(target_id=1, rcs=1.5, gbad=get_direct_factory())
        controller = factory.to_controller(srtm, is_blue=True)
        self.assertIsInstance(controller.effector, DirectFireEffector)
        self.assertEqual(controller.info, TargetInfos.GBAD.with_party(Party.BLUE))

    def test_to_controller_indirect(self):
        factory = StaticGbadFactory(target_id=1, rcs=1.5, gbad=get_indirect_factory())
        controller = factory.to_controller(srtm, is_blue=False)
        self.assertIsInstance(controller.effector, IndirectFireEffector)
        self.assertEqual(controller.info, TargetInfos.GBAD.with_party(Party.RED))

    def test_name_reaches_the_ground_truth(self):
        factory = StaticGbadFactory(
            target_id=1, rcs=1.5, gbad=get_direct_factory(), name="SHORAD"
        )
        controller = factory.to_controller(srtm, is_blue=True)
        controller.update(
            get_empty_situational_picture(), datetime.timedelta(seconds=1)
        )
        [target] = controller.targets
        self.assertEqual(target.name, "SHORAD")

    def test_geojson_range_altitudes_reach_the_controller(self):
        factory = StaticGbadFactory(
            target_id=1,
            rcs=1.5,
            gbad=get_direct_factory(),
            geojson_range_altitudes=[2000, 1000],
        )
        controller = factory.to_controller(srtm, is_blue=True)
        self.assertEqual(controller.geojson_range_altitudes, {1000, 2000})

    def test_geojson_range_altitudes_default_to_none(self):
        factory = StaticGbadFactory(target_id=1, rcs=1.5, gbad=get_direct_factory())
        self.assertEqual(factory.geojson_range_altitudes, set())

    def test_geojson_range_altitudes_survive_a_json_round_trip(self):
        factory = StaticGbadFactory(
            target_id=1,
            rcs=1.5,
            gbad=get_direct_factory(),
            geojson_range_altitudes={1000},
        )
        restored = StaticGbadFactory.model_validate_json(factory.model_dump_json())
        self.assertEqual(restored.geojson_range_altitudes, {1000})

    @staticmethod
    def _gbad_with_clearance(clearance: float) -> StaticGbadFactory:
        """GBAD `clearance` metres above the stub terrain at `p.alt`."""
        gbad = get_direct_factory()
        gbad = gbad.model_copy(
            update={"point": p.model_copy(update={"alt": p.alt + clearance})}
        )
        return StaticGbadFactory(target_id=1, rcs=1.5, gbad=gbad, name="Ground GBAD")

    def test_to_controller_warns_if_on_or_below_terrain(self):
        # Stub terrain at p.alt: no SRTM needed.
        terrain = ConstantSphereTerrainModel(alt=p.alt)
        for clearance in (0.0, -10.0):  # on and below the terrain
            with self.subTest(clearance=clearance):
                factory = self._gbad_with_clearance(clearance)
                with self.assertLogs(level=logging.WARNING) as logs:
                    factory.to_controller(terrain, is_blue=True)
                self.assertEqual(len(logs.output), 1)
                self.assertIn("Ground GBAD", logs.output[0])
                self.assertIn("line-of-sight", logs.output[0])

    def test_to_controller_does_not_warn_if_above_terrain(self):
        terrain = ConstantSphereTerrainModel(alt=p.alt)
        factory = self._gbad_with_clearance(5.0)
        with self.assertNoLogs(level=logging.WARNING):
            factory.to_controller(terrain, is_blue=True)

    def test_discriminated_union_parses_direct_and_indirect(self):
        direct = StaticGbadFactory.model_validate(
            {
                "target_id": 1,
                "rcs": 1.0,
                "gbad": {
                    "type": "direct",
                    "id": 1,
                    "name": "g",
                    "point": p.model_dump(),
                    "combat_range": 100,
                    "n_attacks_left": 1,
                    "cadence": 1.0,
                },
            }
        )
        indirect = StaticGbadFactory.model_validate(
            {
                "target_id": 2,
                "rcs": 1.0,
                "gbad": {
                    "type": "indirect",
                    "id": 2,
                    "name": "i",
                    "point": p.model_dump(),
                    "combat_range": 100,
                    "n_attacks_left": 1,
                    "cadence": 1.0,
                    "projectile": {
                        "id": 3,
                        "name": "proj",
                        "point": p.model_dump(),
                        "combat_range": 50,
                        "n_attacks_left": 1,
                        "cadence": 1.0,
                    },
                    "projectile_speed": 300.0,
                    "projectile_max_dist": 5_000.0,
                    "projectile_rcs": {"rcs": 0.1},
                },
            }
        )
        self.assertIsInstance(direct.gbad, DirectFireEffectorFactory)
        self.assertIsInstance(indirect.gbad, IndirectFireEffectorFactory)

    def test_legacy_gbad_without_type_defaults_to_direct(self):
        # Scenario files saved before indirect fire existed have no "type" key.
        factory = StaticGbadFactory.model_validate(
            {
                "target_id": 1,
                "rcs": 1.0,
                "gbad": {
                    "id": 1,
                    "name": "legacy",
                    "point": p.model_dump(),
                    "combat_range": 100,
                    "n_attacks_left": 1,
                    "cadence": 1.0,
                },
            }
        )
        self.assertIsInstance(factory.gbad, DirectFireEffectorFactory)


class OrderOfBattleGbadsTest(unittest.TestCase):
    def test_update_id_provider_registers_launcher_id_only(self):
        # The projectile is only a template. Each shot deep-copies it and
        # mints its own id.
        oob = get_order_of_battle(
            [StaticGbadFactory(target_id=1, rcs=1.0, gbad=get_indirect_factory())]
        )
        id_provider = IdProvider()
        oob.update_id_provider(id_provider)
        self.assertEqual(id_provider._used_ids[Entity.EFFECTOR], {2})

    def test_reindex_reindexes_launcher_id_only(self):
        oob = get_order_of_battle(
            [StaticGbadFactory(target_id=1, rcs=1.0, gbad=get_indirect_factory())]
        )
        id_provider = IdProvider()
        oob.reindex(id_provider)
        gbad = oob.gbads[0].gbad
        self.assertIsInstance(gbad, IndirectFireEffectorFactory)
        self.assertEqual(gbad.id, 0)
        # The projectile template's id is left as authored, untouched.
        self.assertEqual(gbad.projectile.id, 3)


class CriticalInfrastructureFactoryTest(unittest.TestCase):
    def test_to_controller_blue(self):
        factory = CriticalInfrastructureFactory(target_id=1, name="Airport", point=p)
        living = factory.to_controller(is_blue=True)
        self.assertIsInstance(living, LivingController)
        self.assertEqual(living.target_id, 1)
        self.assertIsInstance(living.child, CriticalInfrastructureController)
        self.assertEqual(
            living.child.info,
            TargetInfos.CRITICAL_INFRASTRUCTURE.with_party(Party.BLUE),
        )
        self.assertEqual(living.child.rcs, 100.0)

    def test_to_controller_red(self):
        factory = CriticalInfrastructureFactory(target_id=1, name="Airport", point=p)
        living = factory.to_controller(is_blue=False)
        self.assertEqual(
            living.child.info,
            TargetInfos.CRITICAL_INFRASTRUCTURE.with_party(Party.RED),
        )

    def test_rcs_defaults_when_omitted_from_file(self):
        factory = CriticalInfrastructureFactory.model_validate(
            {"target_id": 1, "name": "Airport", "point": p.model_dump()}
        )
        self.assertEqual(factory.rcs, 100.0)

    def test_rcs_is_mandatory_on_controller(self):
        with self.assertRaises(Exception):
            CriticalInfrastructureController(
                target_id=1,
                name="Airport",
                point=p,
                info=TargetInfos.CRITICAL_INFRASTRUCTURE.with_party(Party.BLUE),
            )


class OrderOfBattleCriticalInfrastructureTest(unittest.TestCase):
    def test_update_id_provider_registers_target_id(self):
        oob = get_order_of_battle(
            [], [CriticalInfrastructureFactory(target_id=5, name="Airport", point=p)]
        )
        id_provider = IdProvider()
        oob.update_id_provider(id_provider)
        self.assertEqual(id_provider._used_ids[Entity.TARGET], {5})

    def test_reindex_reindexes_target_id(self):
        oob = get_order_of_battle(
            [], [CriticalInfrastructureFactory(target_id=5, name="Airport", point=p)]
        )
        id_provider = IdProvider()
        oob.reindex(id_provider)
        self.assertEqual(oob.critical_infrastructure[0].target_id, 0)


class SimulationResultsAddTest(unittest.TestCase):
    def test_add_concatenates_both_lists(self):
        a = SimulationResults.model_construct(
            monostaticCoverages=["a1"], pclMinDetectableRcsGrids=["a2"]
        )
        b = SimulationResults.model_construct(
            monostaticCoverages=["b1"], pclMinDetectableRcsGrids=["b2"]
        )
        result = a + b
        self.assertEqual(result.monostaticCoverages, ["a1", "b1"])
        self.assertEqual(result.pclMinDetectableRcsGrids, ["a2", "b2"])

    def test_add_does_not_mutate_operands(self):
        a = SimulationResults.model_construct(
            monostaticCoverages=["a1"], pclMinDetectableRcsGrids=[]
        )
        b = SimulationResults.model_construct(
            monostaticCoverages=["b1"], pclMinDetectableRcsGrids=[]
        )
        a + b
        self.assertEqual(a.monostaticCoverages, ["a1"])
        self.assertEqual(b.monostaticCoverages, ["b1"])


def get_coverage(sensor_id: int, x0: float) -> MonostaticCoverageCalculation:
    """Coverage of two separate squares starting at longitude `x0`."""
    return MonostaticCoverageCalculation(
        sensorId=sensor_id,
        settings=MonostaticCoverageCalcSettings(
            targetAlt=1000.0,
            targetRcs=1.5,
            probabilityThreshold=0.5,
            latRes=0.01,
            lonRes=0.01,
        ),
        coverage=[
            GeoJSONFeature.from_shapely(shapely.box(x0, 47.0, x0 + 1.0, 48.0)),
            GeoJSONFeature.from_shapely(shapely.box(x0 + 2.0, 47.0, x0 + 3.0, 48.0)),
        ],
    )


def get_simulation_results(*sensor_ids: int) -> SimulationResults:
    t = datetime.datetime.fromtimestamp(0, datetime.UTC)
    return SimulationResults(
        monostaticCoverages=[(get_coverage(i, 8.0), t) for i in sensor_ids],
        pclMinDetectableRcsGrids=[],
    )


class SimulationResultsToAnnotationsTest(unittest.TestCase):
    def test_coverage_becomes_one_sensor_range_tied_to_radar(self):
        [annotation] = get_simulation_results(3).to_annotations(3, 12, "Radar Uto")

        self.assertEqual(annotation.id, "sensor-range-3-1000-1.5")
        self.assertEqual(annotation.kind, GeoJSONKind.SENSOR_RANGE)
        self.assertEqual(annotation.target_ids, [12])
        self.assertEqual(
            annotation.hover_text, "Radar Uto coverage @ 1000 MASL, RCS=1.5m^2"
        )
        # Both squares of the coverage end up in a single multi-polygon.
        self.assertEqual(annotation.feature.geometry.type, "MultiPolygon")
        self.assertEqual(len(annotation.feature.geometry.coordinates), 2)

    def test_only_coverages_of_the_given_sensor(self):
        results = get_simulation_results(3, 4, 3)

        self.assertEqual(len(results.to_annotations(3, 12)), 2)
        self.assertEqual(len(results.to_annotations(4, 13)), 1)
        self.assertEqual(results.to_annotations(5, 14), [])

    def test_hover_text_falls_back_to_sensor_id(self):
        [annotation] = get_simulation_results(3).to_annotations(3, 12)

        self.assertEqual(
            annotation.hover_text, "Monostatic #3 coverage @ 1000 MASL, RCS=1.5m^2"
        )


def get_empty_situational_picture() -> SituationalPicture:
    return SituationalPicture(
        time=datetime.datetime.fromtimestamp(0, datetime.UTC),
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[],
    )


class OrderOfBattleCoverageTest(unittest.TestCase):
    def setUp(self):
        self._radar = get_uetliberg_radar()
        self._oob = get_order_of_battle([])
        self._oob.monostatic_sensors = [
            MonostaticSensorFactory(target_id=12, rcs=10.0, sensor=self._radar)
        ]
        # The coverage of sensor 99 belongs to no radar of the ORBAT.
        self._oob.simulationResults = get_simulation_results(self._radar.id, 99)

    def get_geojson(self) -> list:
        root = self._oob.to_controller(ConstantSphereTerrainModel(alt=0.0), True)
        root.update(get_empty_situational_picture(), datetime.timedelta(seconds=1))
        return list(root.geojson)

    def test_coverage_is_published_by_its_radar(self):
        [annotation] = self.get_geojson()

        self.assertEqual(annotation.target_ids, [12])

    def test_name_reaches_the_ground_truth(self):
        self._oob.monostatic_sensors[0].name = "Radar Uetliberg"
        root = self._oob.to_controller(ConstantSphereTerrainModel(alt=0.0), True)
        root.update(get_empty_situational_picture(), datetime.timedelta(seconds=1))

        [target] = root.targets
        self.assertEqual(target.name, "Radar Uetliberg")

    def test_user_controlled_radar_publishes_its_coverage(self):
        self._oob.is_user_controlled = True

        [annotation] = self.get_geojson()

        self.assertEqual(annotation.target_ids, [12])


class OrderOfBattleMergeTest(unittest.TestCase):
    def test_merge_does_not_raise(self):
        # Regression test: SimulationResults previously had no __add__, so
        # every call to merge() raised TypeError.
        orbat1 = get_order_of_battle([])
        orbat2 = get_order_of_battle([])
        OrderOfBattle.merge(orbat1, orbat2)

    def test_merge_combines_ballistic_missiles_from_both_orbats(self):
        missile1 = get_ballistic_missile_factory(
            target_id=1, effector_id=2, rcs=1.5, alpha=30.0
        )
        missile2 = get_ballistic_missile_factory(
            target_id=10, effector_id=11, rcs=2.5, alpha=60.0
        )
        orbat1 = get_order_of_battle([], ballistic_missiles=[missile1])
        orbat2 = get_order_of_battle(
            [],
            ballistic_missiles=[missile2],
            oneway_drones=[get_oneway_drone_factory(20)],
        )

        # merge()/reindex() mutate orbat2's objects in place, so snapshot
        # the pre-merge field values instead of comparing against the
        # (by-then-mutated) missile1/missile2 objects themselves.
        missile1_before = missile1.model_dump()
        missile2_before = missile2.model_dump()

        result = OrderOfBattle.merge(orbat1, orbat2)

        self.assertEqual(len(result.ballistic_missiles), 2)
        merged1, merged2 = result.ballistic_missiles

        def without_ids(dump: dict) -> dict:
            return {
                k: v for k, v in dump.items() if k not in ("target_id", "effector_id")
            }

        # orbat1's missile is carried over unchanged, including its ids.
        self.assertEqual(merged1.model_dump(), missile1_before)

        # orbat2's missile keeps every field (start/stop points, launch time,
        # terrain, rcs, alpha) except its ids, which get reindexed.
        self.assertEqual(
            without_ids(merged2.model_dump()), without_ids(missile2_before)
        )

        # Reindexed ids must not collide with orbat1's (or each other's).
        # This means there must be no duplicate IDs.
        target_ids = [merged1.target_id, merged2.target_id]
        effector_ids = [merged1.effector_id, merged2.effector_id]
        self.assertEqual(len(target_ids), len(set(target_ids)))
        self.assertEqual(len(effector_ids), len(set(effector_ids)))

        # Regression check for the copy-paste bug that used to merge in
        # orbat2's drones instead of its missiles.
        self.assertTrue(
            all(
                isinstance(m, BallisticMissileFactory)
                for m in result.ballistic_missiles
            )
        )
        self.assertEqual(len(result.oneway_drones), 1)
        self.assertIsInstance(result.oneway_drones[0], FixedPathOneWayDroneFactory)

    def test_merge_combines_critical_infrastructure_from_both_orbats(self):
        orbat1 = get_order_of_battle(
            [],
            critical_infrastructure=[
                CriticalInfrastructureFactory(target_id=1, name="Airport 1", point=p)
            ],
        )
        orbat2 = get_order_of_battle(
            [],
            critical_infrastructure=[
                CriticalInfrastructureFactory(target_id=2, name="Airport 2", point=p)
            ],
        )
        result = OrderOfBattle.merge(orbat1, orbat2)
        self.assertEqual(len(result.critical_infrastructure), 2)
        self.assertEqual(
            {c.name for c in result.critical_infrastructure},
            {"Airport 1", "Airport 2"},
        )

    def test_merge_reindexes_second_orbats_ids_without_collision(self):
        orbat1 = get_order_of_battle(
            [],
            critical_infrastructure=[
                CriticalInfrastructureFactory(target_id=1, name="Airport 1", point=p)
            ],
            ballistic_missiles=[get_ballistic_missile_factory(2, 3)],
        )
        orbat2 = get_order_of_battle(
            [],
            critical_infrastructure=[
                CriticalInfrastructureFactory(target_id=1, name="Airport 2", point=p)
            ],
            ballistic_missiles=[get_ballistic_missile_factory(2, 3)],
        )
        result = OrderOfBattle.merge(orbat1, orbat2)
        target_ids = [c.target_id for c in result.critical_infrastructure] + [
            m.target_id for m in result.ballistic_missiles
        ]
        self.assertEqual(len(target_ids), len(set(target_ids)))
        effector_ids = [m.effector_id for m in result.ballistic_missiles]
        self.assertEqual(len(effector_ids), len(set(effector_ids)))


class BallisticMissileFactoryCategoryTest(unittest.TestCase):
    def test_default_category(self):
        trajectory = get_ballistic_missile_factory(1, 2).get_trajectory(is_blue=False)
        self.assertEqual(
            trajectory.target_info.category,
            TargetCategory.SHORT_RANGE_BALLISTIC_MISSILE,
        )
        self.assertEqual(trajectory.target_info.party, Party.RED)

    def test_authored_category(self):
        factory = BallisticMissileFactory.model_validate(
            {
                **get_ballistic_missile_factory(1, 2).model_dump(mode="json"),
                "category": "CRUISE_MISSILE",
            }
        )
        trajectory = factory.get_trajectory(is_blue=True)
        self.assertEqual(trajectory.target_info.category, TargetCategory.CRUISE_MISSILE)
        self.assertEqual(trajectory.target_info.party, Party.BLUE)


class BallisticMissileFactoryTimeTest(unittest.TestCase):
    def test_trajectory_starts_at_launch_time(self):
        factory = get_ballistic_missile_factory(1, 2)
        trajectory = factory.get_trajectory(is_blue=True)
        self.assertEqual(trajectory.times[0], factory.t_start)
        self.assertEqual(len(trajectory.times), len(factory._times))
        # timedelta has a resolution of 1 µs.
        for time, seconds in zip(trajectory.times, factory._times):
            self.assertAlmostEqual(
                (time - factory.t_start).total_seconds(), seconds, delta=1e-6
            )


class CruiseMissileFactoryTest(unittest.TestCase):
    def test_default_category_and_party(self):
        factory = get_cruise_missile_factory(1, 2)
        for is_blue, party in [(True, Party.BLUE), (False, Party.RED)]:
            info = factory.get_trajectory(is_blue).target_info
            self.assertEqual(info.category, TargetCategory.CRUISE_MISSILE)
            self.assertEqual(info.party, party)

    def test_trajectory_starts_at_t_start_above_terrain(self):
        factory = get_cruise_missile_factory(1, 2)
        trajectory = factory.get_trajectory(is_blue=False)
        self.assertEqual(trajectory.times[0], factory.t_start)
        self.assertEqual(trajectory.target_id, 1)
        self.assertEqual(trajectory.cross_section_model.rcs, 0.1)
        # At least cruise_magl above ground; here higher, because the terrain
        # right after the launch rises faster than the missile can climb.
        self.assertGreaterEqual(
            trajectory.alts[0], srtm.elevationAt(47.0, 8.0) + 80.0 - 1e-6
        )

    def test_trajectory_ends_at_impact(self):
        factory = get_cruise_missile_factory(1, 2)
        trajectory = factory.get_trajectory(is_blue=False)
        impact = factory.path.impact
        self.assertAlmostEqual(impact.lat, 47.1)
        self.assertAlmostEqual(impact.lon, 8.1)
        self.assertAlmostEqual(impact.alt, srtm.elevationAt(47.1, 8.1), places=6)
        self.assertAlmostEqual(trajectory.lats[-1], impact.lat)
        self.assertAlmostEqual(trajectory.alts[-1], impact.alt)
        self.assertEqual(
            trajectory.times[-1],
            factory.t_start + datetime.timedelta(seconds=factory.path.t_impact + 1.0),
        )

    def test_to_controller(self):
        factory = get_cruise_missile_factory(1, 2)
        living = factory.to_controller(is_blue=True)
        self.assertIsInstance(living, LivingController)
        self.assertEqual(living.target_id, 1)
        missile = living.child
        self.assertIsInstance(missile, FixedPathOneWayDrone)
        self.assertEqual(missile.effector.id, 2)
        self.assertEqual(missile.effector.n_attacks_left, 1)
        self.assertEqual(missile.assigned_goal, factory.path.impact)
        # From the impact until the end of the hold, the missile meets the
        # controller's firing condition (line of sight to its goal, within
        # combat range), so any 1 s simulation time step lets it attack.
        impact_time = factory.t_start + datetime.timedelta(
            seconds=factory.path.t_impact
        )
        self.assertGreaterEqual(
            missile.trajectory.times[-1] - impact_time,
            datetime.timedelta(seconds=1),
        )
        for offset in np.linspace(0.0, 1.0, 21):
            t = impact_time + datetime.timedelta(seconds=float(offset))
            with self.subTest(offset=offset):
                target = missile.trajectory(t)
                self.assertIsNotNone(target)
                self.assertLessEqual(
                    line_of_sight_distance(
                        *target.point.as_tuple(), *missile.assigned_goal.as_tuple()
                    ),
                    missile.effector.combat_range,
                )
                self.assertTrue(
                    missile.terrain.has_line_of_sight(
                        target.point, missile.assigned_goal
                    )
                )

    def test_json_round_trip(self):
        factory = get_cruise_missile_factory(
            1, 2, min_flight_path_angle=-5.0, max_flight_path_angle=20.0
        )
        restored = CruiseMissileFactory.model_validate(factory.model_dump(mode="json"))
        self.assertEqual(restored, factory)

    def test_defaults(self):
        factory = get_cruise_missile_factory(1, 2)
        self.assertEqual(factory.min_flight_path_angle, -10.0)
        self.assertEqual(factory.max_flight_path_angle, 15.0)
        self.assertEqual(factory.terminal_dive_angle, -30.0)
        self.assertEqual(factory.sample_spacing, 100.0)
        self.assertIsNone(factory.min_clearance)
        self.assertEqual(factory.effective_min_clearance, 40.0)
        self.assertEqual(
            get_cruise_missile_factory(
                1, 2, min_clearance=20.0
            ).effective_min_clearance,
            20.0,
        )

    def test_invalid_parameters_rejected(self):
        for kwargs in [
            {"min_flight_path_angle": 5.0},
            {"max_flight_path_angle": -5.0},
            {"terminal_dive_angle": 30.0},
            {"min_clearance": 100.0},
            {"sample_spacing": 0.0},
        ]:
            with self.subTest(**kwargs), self.assertRaises(pydantic.ValidationError):
                get_cruise_missile_factory(1, 2, **kwargs)


class OrderOfBattleCruiseMissileTest(unittest.TestCase):
    def test_file_without_cruise_missiles_loads(self):
        data = get_order_of_battle([]).model_dump(mode="json")
        del data["cruise_missiles"]
        self.assertEqual(OrderOfBattle.model_validate(data).cruise_missiles, [])

    def test_update_id_provider_registers_ids(self):
        orbat = get_order_of_battle(
            [], cruise_missiles=[get_cruise_missile_factory(5, 7)]
        )
        id_provider = IdProvider()
        orbat.update_id_provider(id_provider)
        self.assertIn(5, id_provider._used_ids[Entity.TARGET])
        self.assertEqual(6, id_provider._free_ids[Entity.TARGET])
        self.assertIn(7, id_provider._used_ids[Entity.EFFECTOR])
        self.assertEqual(8, id_provider._free_ids[Entity.EFFECTOR])

    def test_reindex(self):
        orbat = get_order_of_battle(
            [], cruise_missiles=[get_cruise_missile_factory(5, 7)]
        )
        id_provider = IdProvider()
        id_provider.register_entity(Entity.TARGET, 5)
        id_provider.register_entity(Entity.EFFECTOR, 7)
        orbat.reindex(id_provider)
        self.assertEqual(orbat.cruise_missiles[0].target_id, 6)
        self.assertEqual(orbat.cruise_missiles[0].effector_id, 8)

    def test_merge(self):
        orbat1 = get_order_of_battle(
            [], cruise_missiles=[get_cruise_missile_factory(1, 1)]
        )
        orbat2 = get_order_of_battle(
            [], cruise_missiles=[get_cruise_missile_factory(1, 1)]
        )
        merged = OrderOfBattle.merge(orbat1, orbat2)
        self.assertEqual(len(merged.cruise_missiles), 2)
        target_ids = [m.target_id for m in merged.cruise_missiles]
        effector_ids = [m.effector_id for m in merged.cruise_missiles]
        self.assertEqual(len(set(target_ids)), 2)
        self.assertEqual(len(set(effector_ids)), 2)

    def test_time_span(self):
        factory = get_cruise_missile_factory(1, 2)
        orbat = get_order_of_battle([], cruise_missiles=[factory])
        trajectory = factory.get_trajectory(True)
        self.assertEqual(orbat.t_min, trajectory.times[0])
        self.assertEqual(orbat.t_max, trajectory.times[-1])
        self.assertIsNone(get_order_of_battle([]).t_min)
        self.assertIsNone(get_order_of_battle([]).t_max)

    def test_to_controller_includes_cruise_missiles(self):
        orbat = get_order_of_battle(
            [], cruise_missiles=[get_cruise_missile_factory(1, 2)]
        )
        controller = orbat.to_controller(srtm, is_blue=True)
        [tree] = controller._controllers
        children = tree.child._controllers
        self.assertTrue(
            any(
                isinstance(c, LivingController)
                and isinstance(c.child, FixedPathOneWayDrone)
                and c.target_id == 1
                and c.child.trajectory.target_info.category
                == TargetCategory.CRUISE_MISSILE
                for c in children
            )
        )


class FixedPathOneWayDroneFactoryTest(unittest.TestCase):
    def test_party_follows_side(self):
        factory = get_oneway_drone_factory(1)
        for is_blue, party in [(True, Party.BLUE), (False, Party.RED)]:
            living = factory.to_controller(srtm, is_blue)
            info = living.child.trajectory.target_info
            self.assertEqual(info.party, party)
            self.assertEqual(info.category, TargetCategory.DRONE_CLASS_I)
        # The authored factory itself stays untouched.
        self.assertEqual(factory.trajectory.target_info.party, Party.UNKNOWN)


def get_camp_site(
    site_id: int,
    target_id: int,
    ammo=None,
    n_ammo: int = 5,
    can_launch: bool = True,
):
    return CampSite(
        id=site_id,
        name=f"site {site_id}",
        point=p,
        target_id=target_id,
        rcs=10.0,
        n_ammo=n_ammo,
        ammo=OneWayDroneTemplate(rcs=0.1, speed=50.0) if ammo is None else ammo,
        can_launch=can_launch,
        seconds_between_launches=10.0,
    )


def get_game_orbat() -> OrderOfBattle:
    orbat = get_order_of_battle([])
    orbat.camp_sites = [
        get_camp_site(1, 20),
        get_camp_site(2, 21, can_launch=False, n_ammo=50),
    ]
    orbat.transport_times = [
        TransportTime(
            source_site_id=2,
            target_site_id=1,
            seconds_until_leave=60.0,
            seconds_underway=600.0,
        )
    ]
    orbat.is_user_controlled = True
    return orbat


class OrderOfBattleCampSitesTest(unittest.TestCase):
    def test_legacy_orbat_loads_with_defaults(self):
        data = get_order_of_battle([]).model_dump(mode="json")
        for key in ("camp_sites", "transport_times", "is_user_controlled"):
            del data[key]
        orbat = OrderOfBattle.model_validate(data)
        self.assertEqual(orbat.camp_sites, [])
        self.assertEqual(orbat.transport_times, [])
        self.assertFalse(orbat.is_user_controlled)

    def test_round_trip(self):
        orbat = get_game_orbat()
        orbat.camp_sites.append(
            get_camp_site(3, 22, ammo=BallisticMissileTemplate(rcs=1.0))
        )
        orbat.camp_sites.append(
            get_camp_site(4, 23, ammo=CruiseMissileTemplate(rcs=0.1, speed=250.0))
        )
        parsed = OrderOfBattle.model_validate_json(orbat.model_dump_json())
        self.assertEqual(parsed, orbat)
        self.assertEqual(
            [type(s.ammo) for s in parsed.camp_sites],
            [
                OneWayDroneTemplate,
                OneWayDroneTemplate,
                BallisticMissileTemplate,
                CruiseMissileTemplate,
            ],
        )
        self.assertTrue(parsed.is_user_controlled)

    def test_camp_sites_without_user_control_warn(self):
        data = get_game_orbat().model_dump(mode="json")
        data["is_user_controlled"] = False
        with self.assertLogs(level=logging.WARNING) as logs:
            OrderOfBattle.model_validate(data)
        [message] = logs.output
        self.assertIn("2 camp site(s)", message)
        self.assertIn("not user-controlled", message)

    def test_no_warning_if_user_controlled_or_without_camp_sites(self):
        for data in (
            get_game_orbat().model_dump(mode="json"),
            get_order_of_battle([]).model_dump(mode="json"),
        ):
            with self.subTest(is_user_controlled=data["is_user_controlled"]):
                with self.assertNoLogs(level=logging.WARNING):
                    OrderOfBattle.model_validate(data)

    def test_duplicate_site_ids_are_rejected(self):
        data = get_game_orbat().model_dump(mode="json")
        data["camp_sites"][1]["id"] = 1
        with self.assertRaises(pydantic.ValidationError):
            OrderOfBattle.model_validate(data)

    def test_route_to_unknown_site_is_rejected(self):
        data = get_game_orbat().model_dump(mode="json")
        data["transport_times"][0]["target_site_id"] = 99
        with self.assertRaises(pydantic.ValidationError):
            OrderOfBattle.model_validate(data)

    def test_negative_stock_is_rejected(self):
        data = get_game_orbat().model_dump(mode="json")
        data["camp_sites"][0]["n_ammo"] = -1
        with self.assertRaises(pydantic.ValidationError):
            OrderOfBattle.model_validate(data)

    def test_route_between_different_ammo_types_is_rejected(self):
        data = get_game_orbat().model_dump(mode="json")
        data["camp_sites"][0]["ammo"] = {"type": "ballistic_missile", "rcs": 1.0}
        with self.assertRaises(pydantic.ValidationError):
            OrderOfBattle.model_validate(data)

    def test_update_id_provider_registers_site_target_ids(self):
        id_provider = IdProvider()
        get_game_orbat().update_id_provider(id_provider)
        self.assertEqual(id_provider._used_ids[Entity.TARGET], {20, 21})

    def test_update_id_provider_registers_site_ids(self):
        id_provider = IdProvider()
        get_game_orbat().update_id_provider(id_provider)
        self.assertEqual(id_provider._used_ids[Entity.CAMP_SITE], {1, 2})

    def test_reindex_reindexes_site_target_ids(self):
        id_provider = IdProvider()
        id_provider.register_entity(Entity.TARGET, 21)
        orbat = get_game_orbat()
        orbat.reindex(id_provider)
        self.assertEqual([s.target_id for s in orbat.camp_sites], [22, 23])

    def test_reindex_reindexes_site_ids_and_routes(self):
        id_provider = IdProvider()
        id_provider.register_entity(Entity.CAMP_SITE, 2)
        orbat = get_game_orbat()
        orbat.reindex(id_provider)
        self.assertEqual([s.id for s in orbat.camp_sites], [3, 4])
        # The route still leads from the depot (formerly 2) to the launch site
        # (formerly 1).
        self.assertEqual(
            [(r.source_site_id, r.target_site_id) for r in orbat.transport_times],
            [(4, 3)],
        )

    def test_merge_keeps_camp_sites(self):
        merged = OrderOfBattle.merge(get_game_orbat(), get_order_of_battle([]))
        self.assertEqual(len(merged.camp_sites), 2)
        self.assertEqual(len(merged.transport_times), 1)
        self.assertEqual(merged.unused_id_camp_site, 3)

    def test_merge_is_user_controlled_if_either_orbat_is(self):
        game = get_game_orbat()
        plain = get_order_of_battle([])
        self.assertTrue(OrderOfBattle.merge(game, plain).is_user_controlled)
        self.assertTrue(OrderOfBattle.merge(plain, game).is_user_controlled)
        self.assertFalse(OrderOfBattle.merge(plain, plain).is_user_controlled)

    def test_merge_renumbers_camp_sites_of_second_orbat(self):
        merged = OrderOfBattle.merge(get_game_orbat(), get_game_orbat())
        self.assertEqual([s.id for s in merged.camp_sites], [1, 2, 3, 4])
        self.assertEqual(
            [(r.source_site_id, r.target_site_id) for r in merged.transport_times],
            [(2, 1), (4, 3)],
        )
        sites = {s.id: s for s in merged.camp_sites}
        for route in merged.transport_times:
            with self.subTest(route=route):
                self.assertFalse(sites[route.source_site_id].can_launch)
                self.assertTrue(sites[route.target_site_id].can_launch)
        self.assertEqual(merged.unused_id_camp_site, 5)

    def _load(self, data: dict) -> OrderOfBattle:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "orbat.json")
            with open(path, "w") as file:
                json.dump(data, file)
            return OrderOfBattle.from_file(path)

    def test_from_file_accepts_consistent_site_ids(self):
        orbat = get_game_orbat()
        orbat.unused_id_camp_site = 3
        orbat.unused_target_id = 22
        self.assertEqual(self._load(orbat.model_dump(mode="json")), orbat)

    def test_from_file_rejects_used_unused_site_id(self):
        orbat = get_game_orbat()
        orbat.unused_id_camp_site = 2
        orbat.unused_target_id = 22
        with self.assertRaises(ValueError):
            self._load(orbat.model_dump(mode="json"))

    def test_from_file_without_camp_sites_needs_no_site_id(self):
        data = get_order_of_battle([]).model_dump(mode="json")
        del data["unused_id_camp_site"]
        self.assertEqual(self._load(data).unused_id_camp_site, 0)


class AmmoTemplateTest(unittest.TestCase):
    # Properties, since pydantic rejects class attributes without annotation.
    # The concrete templates cover implementing them as ClassVar constants.
    MEMBERS: ClassVar[dict] = {
        "default_path_generator": lambda self: BallisticGenerator(),
        "round_name": property(lambda self: "test round"),
        "round_speed": property(lambda self: 1.0),
        "round_combat_range": property(lambda self: 1.0),
        "round_target_info": property(lambda self: TargetInfos.UNKNOWN),
    }

    def test_base_is_abstract(self):
        with self.assertRaises(TypeError):
            _AmmoTemplateBase(rcs=1.0)

    def test_complete_subclass_can_be_instantiated(self):
        template_class = type("Complete", (_AmmoTemplateBase,), dict(self.MEMBERS))
        self.assertEqual(template_class(rcs=1.0).round_name, "test round")

    def test_subclass_missing_a_member_cannot_be_instantiated(self):
        for missing in self.MEMBERS:
            with self.subTest(missing=missing):
                members = {k: v for k, v in self.MEMBERS.items() if k != missing}
                template_class = type("Incomplete", (_AmmoTemplateBase,), members)
                with self.assertRaises(TypeError):
                    template_class(rcs=1.0)


class TerminationTest(unittest.TestCase):
    def get_scenario(self, **kwargs):
        return build_scenario(
            "test",
            TerrainFactory(terrain_name="SRTM"),
            0,
            get_order_of_battle(
                [], ballistic_missiles=[get_ballistic_missile_factory(1, 1)]
            ),
            get_order_of_battle([]),
            **kwargs,
        )

    def test_absent_means_stop_time(self):
        scenario = self.get_scenario()
        self.assertIsNone(scenario.termination)
        data = scenario.model_dump(mode="json")
        del data["termination"]
        scenario = ScenarioFactory.model_validate(data)
        criterion = scenario.termination_criterion()
        self.assertIsInstance(criterion, TimeCriterion)
        self.assertEqual(criterion._end_time, scenario.stop_time)

    def test_time_and_never(self):
        scenario = self.get_scenario()
        stop_time = datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC)
        data = scenario.model_dump(mode="json")

        data["termination"] = {"type": "time", "stop_time": stop_time.isoformat()}
        parsed = ScenarioFactory.model_validate(data)
        self.assertEqual(parsed.termination, TimeTermination(stop_time=stop_time))
        criterion = parsed.termination_criterion()
        self.assertIsInstance(criterion, TimeCriterion)
        self.assertEqual(criterion._end_time, stop_time)

        data["termination"] = {"type": "never"}
        parsed = ScenarioFactory.model_validate(data)
        self.assertEqual(parsed.termination, NeverTermination())
        self.assertIsInstance(parsed.termination_criterion(), NeverCriterion)
        self.assertEqual(
            ScenarioFactory.model_validate_json(parsed.model_dump_json()).termination,
            NeverTermination(),
        )

    def test_build_game_scenario(self):
        scenario = build_scenario(
            "game",
            TerrainFactory(terrain_name="SRTM"),
            0,
            get_game_orbat(),
            get_order_of_battle([]),
        )
        self.assertEqual(scenario.termination, NeverTermination())

    def test_build_without_mobile_parts_still_raises(self):
        with self.assertRaises(ValueError):
            build_scenario(
                "empty",
                TerrainFactory(terrain_name="SRTM"),
                0,
                get_order_of_battle([]),
                get_order_of_battle([]),
            )


def get_intel_children(controller) -> list[ShooterRevealController]:
    """Intel controllers in the tree built by ``OrderOfBattle.to_controller``."""
    [tree] = controller._controllers
    return [
        c for c in tree.child._controllers if isinstance(c, ShooterRevealController)
    ]


class ShooterRevealFactoryTest(unittest.TestCase):
    def test_to_controller(self):
        factory = ShooterRevealFactory(seconds_valid=90.0)
        for is_blue in (True, False):
            with self.subTest(is_blue=is_blue):
                controller = factory.to_controller(is_blue)
                self.assertIsInstance(controller, ShooterRevealController)
                self.assertEqual(controller.is_blue, is_blue)
                self.assertEqual(
                    controller.valid_time, datetime.timedelta(seconds=90.0)
                )
                self.assertEqual(controller._controllers, [])

    def test_each_controller_is_fresh(self):
        factory = ShooterRevealFactory(seconds_valid=90.0)
        self.assertIsNot(factory.to_controller(True), factory.to_controller(True))

    def test_type_defaults_and_serializes(self):
        factory = ShooterRevealFactory(seconds_valid=90.0)
        self.assertEqual(
            factory.model_dump(mode="json"),
            {"type": "shooter_reveal", "seconds_valid": 90.0},
        )

    def test_seconds_valid_must_be_positive(self):
        for seconds_valid in (0.0, -1.0):
            with self.subTest(seconds_valid=seconds_valid):
                with self.assertRaises(pydantic.ValidationError):
                    ShooterRevealFactory(seconds_valid=seconds_valid)

    def test_seconds_valid_is_mandatory(self):
        with self.assertRaises(pydantic.ValidationError):
            ShooterRevealFactory()


def get_known_target_factory() -> KnownTargetFactory:
    return KnownTargetFactory(
        name="power plant",
        point=p,
        target_info=TargetInfos.CRITICAL_INFRASTRUCTURE.with_party(Party.BLUE),
    )


class KnownTargetFactoryTest(unittest.TestCase):
    def test_to_controller(self):
        factory = get_known_target_factory()
        for is_blue in (True, False):
            with self.subTest(is_blue=is_blue):
                controller = factory.to_controller(is_blue)
                self.assertIsInstance(controller, KnownTargetController)
                self.assertEqual(controller.point, p)
                # Used as given, regardless of the side.
                self.assertEqual(controller.target_info, factory.target_info)

    def test_round_trip(self):
        factory = get_known_target_factory()
        data = factory.model_dump(mode="json")
        self.assertEqual(data["type"], "known_target")
        self.assertEqual(KnownTargetFactory.model_validate(data), factory)

    def test_name_is_optional(self):
        factory = KnownTargetFactory(point=p, target_info=TargetInfos.UNKNOWN)
        self.assertEqual(factory.name, "")

    def test_point_and_target_info_are_mandatory(self):
        for kwargs in ({"point": p}, {"target_info": TargetInfos.UNKNOWN}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(pydantic.ValidationError):
                    KnownTargetFactory(**kwargs)


class OrderOfBattleIntelTest(unittest.TestCase):
    def test_legacy_orbat_loads_without_intel(self):
        data = get_order_of_battle([]).model_dump(mode="json")
        del data["intel"]
        self.assertEqual(OrderOfBattle.model_validate(data).intel, [])

    def test_round_trip(self):
        orbat = get_order_of_battle([])
        orbat.intel = [ShooterRevealFactory(seconds_valid=60.0)]
        parsed = OrderOfBattle.model_validate_json(orbat.model_dump_json())
        self.assertEqual(parsed, orbat)
        self.assertIsInstance(parsed.intel[0], ShooterRevealFactory)

    def test_parses_intel_by_type(self):
        data = get_order_of_battle([]).model_dump(mode="json")
        data["intel"] = [{"type": "shooter_reveal", "seconds_valid": 30}]
        [intel] = OrderOfBattle.model_validate(data).intel
        self.assertEqual(intel, ShooterRevealFactory(seconds_valid=30.0))

    def test_unknown_intel_type_is_rejected(self):
        data = get_order_of_battle([]).model_dump(mode="json")
        for entry in ({"type": "unknown", "seconds_valid": 30}, {"seconds_valid": 30}):
            with self.subTest(entry=entry):
                data["intel"] = [entry]
                with self.assertRaises(pydantic.ValidationError):
                    OrderOfBattle.model_validate(data)

    def test_to_controller_includes_intel(self):
        for is_user_controlled in (False, True):
            for is_blue in (True, False):
                with self.subTest(
                    is_user_controlled=is_user_controlled, is_blue=is_blue
                ):
                    orbat = get_order_of_battle([])
                    orbat.is_user_controlled = is_user_controlled
                    orbat.intel = [ShooterRevealFactory(seconds_valid=60.0)]
                    controller = orbat.to_controller(srtm, is_blue=is_blue)
                    [intel] = get_intel_children(controller)
                    self.assertEqual(intel.is_blue, is_blue)
                    self.assertEqual(intel.valid_time, datetime.timedelta(seconds=60))

    def test_to_controller_without_intel(self):
        controller = get_order_of_battle([]).to_controller(srtm, is_blue=True)
        self.assertEqual(get_intel_children(controller), [])

    def test_enemy_shot_reveals_shooter_through_controller_tree(self):
        t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
        orbat = get_order_of_battle([])
        orbat.intel = [ShooterRevealFactory(seconds_valid=60.0)]
        controller = orbat.to_controller(srtm, is_blue=True)
        controller.bind(SimulationContext(id_provider=IdProvider(), start_time=t0))
        # A shot at a BLUE target was fired by RED.
        shooter = get_direct_factory().to_effector(srtm)
        target = Target(
            id=1,
            is_stationary=True,
            info=TargetInfos.GBAD.with_party(Party.BLUE),
            point=p,
            cross_section_model=ConstantRcsModel(rcs=1.0),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )
        controller.on_event(DirectShot(id=-1, time=t0, shooter=shooter, target=target))
        picture = SituationalPicture(
            time=t0,
            friendly_pet_receivers=[],
            friendly_radars=[],
            friendly_targets=[],
            enemy_targets=[],
        )
        controller.update(picture, datetime.timedelta(seconds=1))
        [sensor] = list(controller.intel_sensors)
        self.assertEqual(sensor.trajectory.lats[0], shooter.point.lat)
        self.assertEqual(sensor.trajectory.lons[0], shooter.point.lon)
        self.assertEqual(sensor.trajectory.alts[0], shooter.point.alt)

    def test_parses_mixed_intel_by_type(self):
        orbat = get_order_of_battle([])
        orbat.intel = [
            ShooterRevealFactory(seconds_valid=60.0),
            get_known_target_factory(),
        ]
        parsed = OrderOfBattle.model_validate_json(orbat.model_dump_json())
        self.assertEqual(parsed, orbat)
        self.assertEqual(
            [type(i) for i in parsed.intel],
            [ShooterRevealFactory, KnownTargetFactory],
        )

    def test_known_target_is_reported_through_controller_tree(self):
        t0 = datetime.datetime.fromtimestamp(0, datetime.UTC)
        orbat = get_order_of_battle([])
        orbat.is_user_controlled = True
        orbat.intel = [get_known_target_factory()]
        controller = orbat.to_controller(srtm, is_blue=False)
        controller.bind(SimulationContext(id_provider=IdProvider(), start_time=t0))
        picture = SituationalPicture(
            time=t0,
            friendly_pet_receivers=[],
            friendly_radars=[],
            friendly_targets=[],
            enemy_targets=[],
        )
        controller.update(picture, datetime.timedelta(seconds=1))
        [sensor] = list(controller.intel_sensors)
        self.assertEqual(sensor.trajectory.lats[0], p.lat)
        self.assertEqual(
            sensor.trajectory.target_info,
            TargetInfos.CRITICAL_INFRASTRUCTURE.with_party(Party.BLUE),
        )

    def test_merge_keeps_intel_of_both_orbats(self):
        orbat1 = get_order_of_battle([])
        orbat1.intel = [ShooterRevealFactory(seconds_valid=10.0)]
        orbat2 = get_order_of_battle([])
        orbat2.intel = [ShooterRevealFactory(seconds_valid=20.0)]
        merged = OrderOfBattle.merge(orbat1, orbat2)
        self.assertEqual(
            [i.seconds_valid for i in merged.intel],
            [10.0, 20.0],
        )


if __name__ == "__main__":
    unittest.main()
