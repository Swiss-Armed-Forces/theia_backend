import datetime
import unittest
from unittest.mock import patch

import numpy as np
import pydantic
import shapely

from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.distance import line_of_sight_distance
from theia.effectors import DirectFireEffector, IndirectFireEffector
from theia.simulation.controllers.static_gbad_controller import StaticGbadController
from theia.terrain import DummyTerrain, SrtmTerrainModel
from theia.types import (
    ConstantRcsModel,
    GeoJSONKind,
    Party,
    Point,
    SituationalPicture,
    Target,
    TargetInfo,
    TargetInfos,
    Track,
    Velocity,
)

srtm = SrtmTerrainModel()

p_uetliberg = Point(
    lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
    lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
    alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
)

p_close = CoordinateTransformations.geodetic_to_cartesian(
    lat=47.37348,
    lon=8.53707,
    alt=1000.0,
)

p_close2 = CoordinateTransformations.geodetic_to_cartesian(
    lat=47.37790,
    lon=8.49466,
    alt=1000.0,
)

t0 = datetime.datetime.fromtimestamp(0)
t1 = datetime.datetime.fromtimestamp(1)


GROUNDED_JET = TargetInfos.FIXED_WING.model_copy(update={"is_airborne": False})
"""A fighter jet on the ground: never to be engaged by a GBAD."""


def get_situational_picture_in_range(
    info: TargetInfo = TargetInfos.FIXED_WING,
) -> SituationalPicture:
    track = Track(
        id="0",
        target_info=info,
        states=[
            (
                t0,
                np.array([p_close[0], 0.0, p_close[1], 0.0, p_close[2], 0.0]),
            ),
            (
                t1,
                np.array([p_close[0], 0.0, p_close[1], 0.0, p_close[2], 0.0]),
            ),
        ],
    )
    # Used to check whether the controller just takes the first track or actually searches.
    red_herring = Track(
        id="1",
        target_info=TargetInfos.FIXED_WING,
        states=[
            (
                t0,
                np.array([p_close2[0], 0.0, p_close2[1], 0.0, p_close2[2], 0.0]),
            ),
            (
                t1,
                np.array([p_close2[0], 0.0, p_close2[1], 0.0, p_close2[2], 0.0]),
            ),
        ],
    )
    return SituationalPicture(
        time=t0,
        friendly_pet_receivers=[],
        friendly_radars=[],
        friendly_targets=[],
        enemy_targets=[red_herring, track],
    )


def get_direct_effector() -> DirectFireEffector:
    return DirectFireEffector(
        id=5,
        point=p_uetliberg,
        combat_range=100_000,
        n_attacks_left=1,
        name="",
        terrain=srtm,
        # Cadence is checked/tracked entirely by the effector's own fire()
        # (see tests/test_effectors.py::CadenceTest) - unlimited here since
        # these tests are about the controller's targeting decision, not
        # cadence.
        cadence=float("inf"),
    )


def get_direct_controller(assigned_track_id: str | None) -> StaticGbadController:
    return StaticGbadController(
        target_id=4,
        info=TargetInfos.GBAD.with_party(Party.BLUE),
        rcs=1.5,
        effector=get_direct_effector(),
        assigned_track_id=assigned_track_id,
    )


def get_indirect_projectile_effector() -> DirectFireEffector:
    return DirectFireEffector(
        id=99,
        point=p_uetliberg,
        combat_range=5_000,
        n_attacks_left=1,
        name="",
        terrain=srtm,
        cadence=float("inf"),
    )


def get_indirect_effector(
    combat_range: float = 100_000,
    n_attacks_left: int = 1,
) -> IndirectFireEffector:
    return IndirectFireEffector(
        id=5,
        point=p_uetliberg,
        combat_range=combat_range,
        n_attacks_left=n_attacks_left,
        name="",
        projectile=get_indirect_projectile_effector(),
        projectile_speed=300.0,
        projectile_max_dist=50_000.0,
        projectile_info=TargetInfos.INTERCEPTOR.with_party(Party.BLUE),
        projectile_rcs=ConstantRcsModel(rcs=0.1),
        # Cadence is checked/tracked entirely by the effector's own fire()
        # (see tests/test_effectors.py::CadenceTest) - unlimited here since
        # these tests are about the controller's targeting decision, not
        # cadence.
        cadence=float("inf"),
    )


def get_indirect_controller(
    assigned_track_id: str | None,
    n_attacks_left: int = 1,
) -> StaticGbadController:
    return StaticGbadController(
        target_id=4,
        info=TargetInfos.GBAD.with_party(Party.BLUE),
        rcs=1.5,
        effector=get_indirect_effector(n_attacks_left=n_attacks_left),
        assigned_track_id=assigned_track_id,
    )


class StaticGbadControllerDirectFireTest(unittest.TestCase):
    def test_no_assigned_target(self):
        controller = get_direct_controller(None)
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))
        fire_decisions = controller.firing_effectors
        self.assertEqual(len(fire_decisions), 0)

    def test_assigned_target(self):
        controller = get_direct_controller("0")
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))

        fire_decisions = controller.firing_effectors
        self.assertEqual(len(fire_decisions), 1)

        effector, p = fire_decisions[0]

        self.assertEqual(effector, controller.effector)
        self.assertAlmostEqual(p.lat, 47.37348)
        self.assertAlmostEqual(p.lon, 8.53707)
        self.assertAlmostEqual(p.alt, 1000.0)

    def test_assigned_ground_track_is_not_engaged(self):
        for info in (TargetInfos.UNKNOWN, GROUNDED_JET):
            with self.subTest(info=info):
                controller = get_direct_controller("0")
                picture = get_situational_picture_in_range(info)
                controller.update(picture, datetime.timedelta(seconds=1))
                self.assertEqual(controller.firing_effectors, [])

    def test_geojson_effector_ranges(self):
        controller = get_direct_controller(None)
        controller.target_name = "GBAD Uetliberg"
        controller.geojson_range_altitudes = {3000, 1000}
        with patch(
            "theia.simulation.controllers.static_gbad_controller.calculate_coverage",
            return_value=[shapely.box(8.0, 47.0, 9.0, 48.0)],
        ):
            controller.update(
                get_situational_picture_in_range(), datetime.timedelta(seconds=1)
            )

        self.assertEqual(
            [(a.id, a.hover_text) for a in controller.geojson],
            [
                ("effector-range-4-1000", "GBAD Uetliberg range @ 1000 MASL"),
                ("effector-range-4-3000", "GBAD Uetliberg range @ 3000 MASL"),
            ],
        )
        for annotation in controller.geojson:
            self.assertEqual(annotation.kind, GeoJSONKind.EFFECTOR_RANGE)
            self.assertEqual(annotation.target_ids, [4])

    def test_geojson_effector_range_is_computed_once_per_altitude(self):
        controller = get_direct_controller(None)
        controller.geojson_range_altitudes = {1000}
        dt = datetime.timedelta(seconds=1)
        with patch(
            "theia.simulation.controllers.static_gbad_controller.calculate_coverage",
            return_value=[shapely.box(8.0, 47.0, 9.0, 48.0)],
        ) as calculate_coverage:
            controller.update(get_situational_picture_in_range(), dt)
            controller.update(get_situational_picture_in_range(), dt)
            controller.geojson_range_altitudes = {1000, 2000}
            controller.update(get_situational_picture_in_range(), dt)

        self.assertEqual(
            [call.args[3] for call in calculate_coverage.call_args_list],
            [1000, 2000],
        )

    def test_geojson_effector_range_with_real_coverage(self):
        # Regression test: calculate_coverage returns a list of polygons.
        controller = StaticGbadController(
            target_id=4,
            info=TargetInfos.GBAD.with_party(Party.BLUE),
            rcs=1.5,
            effector=get_direct_effector().model_copy(
                update={
                    "terrain": DummyTerrain(has_los=True),
                    "combat_range": 5_000,
                }
            ),
            geojson_range_altitudes={1000},
        )
        controller.update(get_situational_picture_in_range(), datetime.timedelta(seconds=1))

        [annotation] = controller.geojson
        self.assertEqual(annotation.feature.geometry.type, "MultiPolygon")
        self.assertGreater(len(annotation.feature.geometry.coordinates), 0)

    def test_unknown_effector_type_raises(self):
        # Bypasses validation, which only admits the known effector types.
        controller = get_direct_controller("0")
        object.__setattr__(controller, "effector", object())
        controller.geojson_range_altitudes = {1000}

        with self.assertRaises(TypeError):
            controller._effector_range(1000)

    def test_geojson_range_altitudes_reject_fractions(self):
        controller = get_direct_controller(None)
        controller.geojson_range_altitudes = [1000.0]
        self.assertEqual(controller.geojson_range_altitudes, {1000})
        with self.assertRaises(pydantic.ValidationError):
            controller.geojson_range_altitudes = [1000.5]

    def test_as_target(self):
        controller = get_direct_controller(None)
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))
        targets = controller.targets

        expected = Target(
            id=4,
            is_stationary=True,
            info=TargetInfos.GBAD.with_party(Party.BLUE),
            point=p_uetliberg,
            cross_section_model=ConstantRcsModel(rcs=1.5),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], expected)


class StaticGbadControllerIndirectFireTest(unittest.TestCase):
    def get_ranges(self, combat_range: float, altitudes: set[int]) -> dict:
        controller = StaticGbadController(
            target_id=4,
            info=TargetInfos.GBAD.with_party(Party.BLUE),
            rcs=1.5,
            effector=get_indirect_effector(combat_range=combat_range),
            target_name="LORAD Uetliberg",
            geojson_range_altitudes=altitudes,
        )
        controller.update(
            get_situational_picture_in_range(), datetime.timedelta(seconds=1)
        )
        return {a.id: a for a in controller.geojson}

    def test_geojson_effector_range_is_at_combat_range(self):
        # No line of sight required: the range is a circle at combat range.
        ranges = self.get_ranges(20_000, {5000})

        annotation = ranges["effector-range-4-5000"]
        self.assertEqual(annotation.kind, GeoJSONKind.EFFECTOR_RANGE)
        self.assertEqual(annotation.target_ids, [4])
        self.assertEqual(annotation.hover_text, "LORAD Uetliberg range @ 5000 MASL")
        [[ring]] = annotation.feature.geometry.coordinates
        for lon, lat in ring:
            d = line_of_sight_distance(
                p_uetliberg.lat, p_uetliberg.lon, p_uetliberg.alt, lat, lon, 5000
            )
            self.assertAlmostEqual(d, 20_000, delta=5)

    def test_geojson_effector_range_shrinks_with_altitude(self):
        ranges = self.get_ranges(20_000, {1000, 15_000})

        low, high = (
            shapely.Polygon(ranges[f"effector-range-4-{alt}"]
                            .feature.geometry.coordinates[0][0])
            for alt in (1000, 15_000)
        )
        self.assertTrue(low.contains(high))

    def test_geojson_effector_range_omits_altitudes_out_of_range(self):
        ranges = self.get_ranges(5_000, {1000, 20_000})

        self.assertEqual(list(ranges.keys()), ["effector-range-4-1000"])

    def test_no_assigned_target(self):
        controller = get_indirect_controller(None)
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))
        self.assertEqual(len(controller.firing_effectors), 0)

    def test_assigned_target_not_present(self):
        controller = get_indirect_controller("5")
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))
        self.assertEqual(len(controller.firing_effectors), 0)

    def test_assigned_target(self):
        controller = get_indirect_controller("0")
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))

        fire_decisions = controller.firing_effectors
        self.assertEqual(len(fire_decisions), 1)

        effector, p = fire_decisions[0]

        self.assertEqual(effector, controller.effector)
        self.assertEqual(effector.id, 5)
        # The effector must carry the track ID it was aimed at, so the
        # Simulator can embed it into the resulting IndirectShot.
        self.assertEqual(effector.assigned_track_id, "0")
        self.assertAlmostEqual(p.lat, 47.37348)
        self.assertAlmostEqual(p.lon, 8.53707)
        self.assertAlmostEqual(p.alt, 1000.0)

    def test_assigned_ground_track_is_not_engaged(self):
        for info in (TargetInfos.UNKNOWN, GROUNDED_JET):
            with self.subTest(info=info):
                controller = get_indirect_controller("0")
                picture = get_situational_picture_in_range(info)
                controller.update(picture, datetime.timedelta(seconds=1))
                self.assertEqual(controller.firing_effectors, [])
                self.assertIsNone(controller.effector.assigned_track_id)

    def test_as_target(self):
        controller = get_indirect_controller(None)
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))
        targets = controller.targets

        expected = Target(
            id=4,
            is_stationary=True,
            info=TargetInfos.GBAD.with_party(Party.BLUE),
            point=p_uetliberg,
            cross_section_model=ConstantRcsModel(rcs=1.5),
            velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], expected)

    def test_as_target_shows_damaged_sidc_when_out_of_ammo(self):
        controller = get_indirect_controller(None, n_attacks_left=0)
        picture = get_situational_picture_in_range()
        controller.update(picture, datetime.timedelta(seconds=1))

        self.assertEqual(
            controller.targets[0].sidc,
            TargetInfos.GBAD.with_party(Party.BLUE).sidc(damaged=True),
        )


if __name__ == "__main__":
    unittest.main()
