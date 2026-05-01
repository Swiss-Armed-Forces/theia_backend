import numpy as np
import pydantic
import shapely


class LatLonHeightGrid(pydantic.BaseModel):
    lat_start: float
    lat_stop: float
    lat_res: float

    lon_start: float
    lon_stop: float
    lon_res: float

    height_start: float
    height_stop: float
    height_res: float

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # Correct stop values to match the resolution.
        n_points_lat = int(np.ceil((self.lat_stop - self.lat_start) / self.lat_res)) + 1
        n_points_lon = int(np.ceil((self.lon_stop - self.lon_start) / self.lon_res)) + 1
        n_points_height = (
            int(np.ceil((self.height_stop - self.height_start) / self.height_res)) + 1
        )

        self.lat_stop = self.lat_start + (n_points_lat - 1) * self.lat_res
        self.lon_stop = self.lon_start + (n_points_lon - 1) * self.lon_res
        self.height_stop = self.height_start + (n_points_height - 1) * self.height_res

    @property
    def n_points_lat(self) -> int:
        return int(np.round((self.lat_stop - self.lat_start) / self.lat_res)) + 1

    @property
    def n_points_lon(self) -> int:
        return int(np.round((self.lon_stop - self.lon_start) / self.lon_res)) + 1

    @property
    def n_points_height(self) -> int:
        return (
            int(np.round((self.height_stop - self.height_start) / self.height_res)) + 1
        )

    @property
    def n_points(self) -> tuple[int, int, int]:
        """Number of points along lat, lon and alt directions."""
        return (self.n_points_lat, self.n_points_lon, self.n_points_height)

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.lat_stop + self.lat_start),
            0.5 * (self.lon_stop + self.lon_start),
            0.5 * (self.height_stop + self.height_start),
        )

    @property
    def latitude_values(self) -> np.ndarray:
        return np.linspace(self.lat_start, self.lat_stop, self.n_points_lat)

    @property
    def longitude_values(self) -> np.ndarray:
        return np.linspace(self.lon_start, self.lon_stop, self.n_points_lon)

    @property
    def altitude_values(self) -> np.ndarray:
        return np.linspace(self.height_start, self.height_stop, self.n_points_height)

    @property
    def points(self) -> np.ndarray:
        """Points in the grid. Shape: ``self.n_points``"""
        n = self.n_points[0] * self.n_points[1] * self.n_points[2]
        points = np.empty((n, 3), dtype=np.float32)
        i = 0
        for lat in self.latitude_values:
            for lon in self.longitude_values:
                for height in self.altitude_values:
                    points[i, :] = lat, lon, height
                    i += 1
        return points

    def get_bbox_polygon(self) -> shapely.Polygon:
        return shapely.Polygon(
            shell=[
                [self.lon_start, self.lat_start],
                [self.lon_start, self.lat_stop],
                [self.lon_stop, self.lat_stop],
                [self.lon_stop, self.lat_start],
            ],
            holes=[],
        )