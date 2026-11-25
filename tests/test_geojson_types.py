import unittest

import shapely

from theia.types import (
    GeoJSONAnnotation,
    GeoJSONFeature,
    GeoJSONKind,
    GeoJSONMultiPolygon,
    GeoJSONPolygon,
)


def square(x0: float) -> shapely.Polygon:
    return shapely.box(x0, 0.0, x0 + 1.0, 1.0)


class GeoJSONMultiPolygonMergeTest(unittest.TestCase):
    def test_merges_polygons_and_multipolygons(self):
        polygon = GeoJSONFeature.from_shapely(square(0.0))
        multi = GeoJSONFeature.from_shapely(
            shapely.MultiPolygon([square(2.0), square(4.0)])
        )

        merged = GeoJSONMultiPolygon.merge([polygon, multi])

        self.assertEqual(len(merged.coordinates), 3)
        self.assertEqual(merged.coordinates[0], polygon.geometry.coordinates)
        self.assertEqual(merged.coordinates[1:], multi.geometry.coordinates)
        expected = shapely.MultiPolygon([square(0.0), square(2.0), square(4.0)])
        self.assertTrue(shapely.geometry.shape(merged.model_dump()).equals(expected))

    def test_merge_of_nothing_is_empty(self):
        self.assertEqual(GeoJSONMultiPolygon.merge([]).coordinates, [])


class GeoJSONAnnotationTest(unittest.TestCase):
    def setUp(self):
        self._feature = GeoJSONFeature(
            geometry=GeoJSONPolygon(coordinates=[[[0.0, 0.0]]])
        )

    def test_defaults(self):
        annotation = GeoJSONAnnotation(id="a", feature=self._feature)

        self.assertEqual(annotation.kind, GeoJSONKind.OTHER)
        self.assertEqual(annotation.target_ids, [])
        self.assertIsNone(annotation.hover_text)

    def test_json_round_trip_serializes_kind_as_string(self):
        annotation = GeoJSONAnnotation(
            id="a",
            feature=self._feature,
            kind=GeoJSONKind.SENSOR_RANGE,
            target_ids=[3],
            hover_text="Radar",
        )

        dumped = annotation.model_dump(mode="json")

        self.assertEqual(dumped["kind"], "SENSOR_RANGE")
        self.assertEqual(GeoJSONAnnotation.model_validate(dumped), annotation)


if __name__ == "__main__":
    unittest.main()
