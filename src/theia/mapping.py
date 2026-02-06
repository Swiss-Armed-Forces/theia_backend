import folium
from shapely.geometry import Polygon, LineString

from theia.coordinates import POSITIONS_OF_INTEREST
from theia.types import Radar, Target, Trajectory


class RadarMap:
    def __init__(
        self,
        radars: dict[str, Radar] = {},
        targets: dict[str, Target] = {},
        polygons: dict[str, Polygon] = {},
        paths: dict[str, LineString] = {},
        trajectories: dict[str, Trajectory] = {},
    ):
        self.radars = radars
        self.targets = targets
        self.polygons = polygons
        self.paths = paths
        self.trajectories = trajectories

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
            if radar.transmitter.point == radar.receiver.point:
                folium.Marker(
                    location=(radar.transmitter.lat, radar.transmitter.lon),
                    tooltip=f"{name} (monostatic)",
                ).add_to(map)
            else:
                folium.Marker(
                    location=(radar.transmitter.lat, radar.transmitter.lon),
                    tooltip=f"{name} (Tx)",
                ).add_to(map)
                folium.Marker(
                    location=(radar.receiver.lat, radar.receiver.lon),
                    tooltip=f"{name} (Rx)",
                ).add_to(map)

        for name, target in self.targets.items():
            folium.Marker(
                location=(target.lat, target.lon),
                tooltip=name,
                icon=folium.Icon(color="red"),
            ).add_to(map)

        for name, path in self.paths.items():
            folium.GeoJson(
                path,
                tooltip=name,
            ).add_to(map)

        for name, polygon in self.polygons.items():
            folium.GeoJson(
                polygon,
                tooltip=name,
            ).add_to(map)
        
        for name, trajectory in self.trajectories.items():
            folium.GeoJson(
                trajectory.to_geojson(),
                tooltip=name,
            ).add_to(map)

        return map
