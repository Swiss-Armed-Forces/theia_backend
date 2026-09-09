import unittest

from theia.config import SIDC
from theia.effectors import DirectFireEffector, IndirectFireEffector
from theia.simulation.controllers.critical_infrastructure_controller import (
    CriticalInfrastructureController,
)
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.scenario_import import (
    CriticalInfrastructureFactory,
    DirectFireEffectorFactory,
    IndirectFireEffectorFactory,
    OrderOfBattle,
    StaticGbadFactory,
)
from theia.terrain import SrtmTerrainModel
from theia.types import ConstantRcsModel, Entity, IdProvider, Point

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
) -> OrderOfBattle:
    return OrderOfBattle(
        monostatic_sensors=[],
        pcl_sensors=[],
        gbads=gbads,
        simulationResults={"monostaticCoverages": [], "pclMinDetectableRcsGrids": []},
        oneway_drones=[],
        ballistic_missiles=[],
        critical_infrastructure=critical_infrastructure or [],
        unused_id_sensor=0,
        unused_id_receiver=0,
        unused_id_transmitter=0,
        unused_id_effector=0,
        unused_target_id=0,
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
        self.assertEqual(effector.projectile_sidc, SIDC.BLUE_MISSILE)

    def test_to_effector_red(self):
        effector = get_indirect_factory().to_effector(srtm, is_blue=False)
        self.assertEqual(effector.projectile_sidc, SIDC.RED_MISSILE)


class StaticGbadFactoryTest(unittest.TestCase):
    def test_to_controller_direct(self):
        factory = StaticGbadFactory(target_id=1, rcs=1.5, gbad=get_direct_factory())
        controller = factory.to_controller(srtm, is_blue=True)
        self.assertIsInstance(controller.effector, DirectFireEffector)
        self.assertEqual(controller.sidc, SIDC.BLUE_AIR_DEFENCE)

    def test_to_controller_indirect(self):
        factory = StaticGbadFactory(target_id=1, rcs=1.5, gbad=get_indirect_factory())
        controller = factory.to_controller(srtm, is_blue=False)
        self.assertIsInstance(controller.effector, IndirectFireEffector)
        self.assertEqual(controller.sidc, SIDC.RED_AIR_DEFENCE)

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
        self.assertEqual(living.child.sidc, SIDC.BLUE_GOVERNMENT_SITE)
        self.assertEqual(living.child.rcs, 100.0)

    def test_to_controller_red(self):
        factory = CriticalInfrastructureFactory(target_id=1, name="Airport", point=p)
        living = factory.to_controller(is_blue=False)
        self.assertEqual(living.child.sidc, SIDC.RED_GOVERNMENT_SITE)

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
                sidc=SIDC.BLUE_GOVERNMENT_SITE,
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


if __name__ == "__main__":
    unittest.main()
