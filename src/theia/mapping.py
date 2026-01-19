import folium
from shapely.geometry.polygon import Polygon

from theia.coordinates import POSITIONS_OF_INTEREST
from theia.types import Radar, Target


class RadarMap:
    def __init__(
        self,
        radars: dict[str, Radar] = {},
        targets: dict[str, Target] = {},
        polygons: dict[str, Polygon] = {},
    ):
        self.radars = radars
        self.targets = targets
        self.polygons = polygons

    def to_map(self) -> folium.folium.Map:
        map = folium.Map(
            zoom_start=10,
            location=(
                POSITIONS_OF_INTEREST["CH_CENTER"]["lat"],
                POSITIONS_OF_INTEREST["CH_CENTER"]["lon"],
            ),
            control_scale=True,
        )

        for name, radar in self.radars.items():
            folium.Marker(
                location=(radar.lat, radar.lon),
                tooltip=name,
            ).add_to(map)

        for name, target in self.targets.items():
            folium.Marker(
                location=(target.lat, target.lon),
                tooltip=name,
                icon=folium.Icon(color="red"),
            ).add_to(map)

        for name, polygon in self.polygons.items():
            folium.GeoJson(
                polygon,
                tooltip=name,
            ).add_to(map)

        return map
