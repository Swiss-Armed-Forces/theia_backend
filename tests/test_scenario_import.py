import datetime
import unittest

import numpy as np
import pydantic

from theia.distance import line_of_sight_distance
from theia.effectors import DirectFireEffector, IndirectFireEffector
from theia.simulation.controllers.fixed_path_kamikaze_drone import FixedPathOneWayDrone
from theia.simulation.controllers.critical_infrastructure_controller import (
    CriticalInfrastructureController,
)
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.scenario_import import (
    BallisticMissileFactory,
    CriticalInfrastructureFactory,
    CruiseMissileFactory,
    DirectFireEffectorFactory,
    FixedPathOneWayDroneFactory,
    IndirectFireEffectorFactory,
    OrderOfBattle,
    SimulationResults,
    StaticGbadFactory,
    TerrainFactory,
    Trajectory,
)
from theia.terrain import SrtmTerrainModel
from theia.types import (
    ConstantRcsModel,
    Entity,
    IdProvider,
    Party,
    Point,
    TargetCategory,
    TargetInfos,
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
            vxs=[0.0, 0.0],
            vys=[0.0, 0.0],
            vzs=[0.0, 0.0],
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
            return {k: v for k, v in dump.items() if k not in ("target_id", "effector_id")}

        # orbat1's missile is carried over unchanged, including its ids.
        self.assertEqual(merged1.model_dump(), missile1_before)

        # orbat2's missile keeps every field (start/stop points, launch time,
        # terrain, rcs, alpha) except its ids, which get reindexed.
        self.assertEqual(without_ids(merged2.model_dump()), without_ids(missile2_before))

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
        children = controller.child._controllers
        self.assertTrue(
            any(
                isinstance(c, LivingController)
                and isinstance(c.child, FixedPathOneWayDrone)
                and c.target_id == 1
                and c.child.trajectory.target_info.category == TargetCategory.CRUISE_MISSILE
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


if __name__ == "__main__":
    unittest.main()
