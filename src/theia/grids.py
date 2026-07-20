from functools import cached_property
import itertools
import math

import numpy as np
import pydantic
import shapely
from scipy.ndimage import maximum_filter
from theia.coordinates import CoordinateTransformations, EcefToEnuTransformer
from theia.terrain import AbstractTerrainModel
from theia.types import Point


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


class LatLonTerrainGrid(pydantic.BaseModel):
    lat_start: float
    lat_stop: float
    lat_res: float

    lon_start: float
    lon_stop: float
    lon_res: float

    terrain_model: AbstractTerrainModel

    def __init__(self, *a, **kw):
        """
        Parameters
        ----------
        lat_start: float
            Start latitude [°]
        lat_stop: float
            Stop latitude [°]
        lat_res: float
            Latitude resolution [°]
        lon_start: float
            Start longitude [°]
        lon_stop: float
            Stop longitude [°]
        lon_res: float
            Longitude resolution [°]
        terrain_model: AbstractTerrainModel
            Terrain model
        """
        super().__init__(*a, **kw)

        # Correct stop values to match the resolution.
        n_points_lat = int(np.ceil((self.lat_stop - self.lat_start) / self.lat_res)) + 1
        n_points_lon = int(np.ceil((self.lon_stop - self.lon_start) / self.lon_res)) + 1

        self.lat_stop = self.lat_start + (n_points_lat - 1) * self.lat_res
        self.lon_stop = self.lon_start + (n_points_lon - 1) * self.lon_res

    @property
    def n_points_lat(self) -> int:
        return int(np.round((self.lat_stop - self.lat_start) / self.lat_res)) + 1

    @property
    def n_points_lon(self) -> int:
        return int(np.round((self.lon_stop - self.lon_start) / self.lon_res)) + 1

    @property
    def n_points(self) -> tuple[int, int, int]:
        """Number of points along lat, lon, alt directions."""
        return (self.n_points_lat, self.n_points_lon, 1)

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.lat_stop + self.lat_start),
            0.5 * (self.lon_stop + self.lon_start),
            self.terrain_model.elevationAt(
                0.5 * (self.lat_stop + self.lat_start),
                0.5 * (self.lon_stop + self.lon_start),
            ),
        )

    @property
    def latitude_values(self) -> np.ndarray:
        return np.linspace(self.lat_start, self.lat_stop, self.n_points_lat)

    @property
    def longitude_values(self) -> np.ndarray:
        return np.linspace(self.lon_start, self.lon_stop, self.n_points_lon)

    @property
    def points(self) -> np.ndarray:
        """Points in the grid. Shape: ``self.n_points``"""
        n = self.n_points[0] * self.n_points[1] * self.n_points[2]
        points = np.empty((n, 3), dtype=np.float32)
        i = 0
        for lat in self.latitude_values:
            for lon in self.longitude_values:
                points[i, :] = lat, lon, self.terrain_model.elevationAt(lat, lon)
                i += 1
        return points

    @property
    def is_local_maximum(self) -> np.ndarray:
        alts = np.empty((self.n_points[0], self.n_points[1]), dtype=np.float32)
        for i, lat in enumerate(self.latitude_values):
            for j, lon in enumerate(self.longitude_values):
                alts[i, j] = self.terrain_model.elevationAt(lat, lon)
        is_max = alts == maximum_filter(alts, size=3)
        return is_max.flatten()

    def get_points_local_maxima(self, radius: int = 3) -> np.ndarray:
        alts = np.empty((self.n_points[0], self.n_points[1]), dtype=np.float32)
        for i, lat in enumerate(self.latitude_values):
            for j, lon in enumerate(self.longitude_values):
                alts[i, j] = self.terrain_model.elevationAt(lat, lon)
        is_max = alts == maximum_filter(alts, size=radius)

        ii, jj = np.where(is_max)
        points = []
        lats = self.latitude_values
        lons = self.longitude_values
        for i, j in zip(ii, jj, strict=True):
            points.append((lats[i], lons[j], alts[i, j]))

        return np.array(points)

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

    def contains(self, point: Point) -> bool:
        return (
            self.lat_start <= point.lat
            and point.lat <= self.lat_stop
            and self.lon_start <= point.lon
            and point.lon <= self.lon_stop
        )


class EnuGridMask(pydantic.BaseModel):
    reference_point: Point
    east_min: float
    """Minimum east coordinate [m]"""
    east_max: float
    """Maximum east coordinate [m]"""
    north_min: float
    """Minimum north coordinate [m]"""
    north_max: float
    """Maximum north coordinate [m]"""
    up_min: float
    """Minimum up coordinate [m]"""
    up_max: float
    """Maximum up coordinate [m]"""
    res_north_east: float
    """Grid resolution in north and east directions [m]"""
    res_up: float
    """Grid resolution in up direction [m]"""
    terrain: AbstractTerrainModel

    @cached_property
    def points(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Points in geodetic coordinates [m]; shape (N_east, N_north, N_up, 3)
        """
        transformer = EcefToEnuTransformer(reference_point=self.reference_point)

        easts = np.arange(self.east_min, self.east_max, self.res_north_east)
        norths = np.arange(self.north_min, self.north_max, self.res_north_east)
        ups = np.arange(self.up_min, self.up_max, self.res_up)

        points = np.empty((len(easts), len(norths), len(ups), 3))
        for i, east in enumerate(easts):
            for j, north in enumerate(norths):
                for k, up in enumerate(ups):
                    points[i, j, k, :] = (
                        CoordinateTransformations.cartesian_to_geodetic(
                            *transformer.enu_to_ecef((north, east, up))
                        )
                    )
        return points

    @property
    def binary_mask(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Binary mask indicating terrain (True) or open space (False);
            shape (n_north, n_east, n_up)
        """
        points = self.points
        mask = np.empty(
            (points.shape[0], points.shape[1], points.shape[2]),
            dtype=bool,
        )
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                for k in range(points.shape[2]):
                    lat = points[i, j, k, 0]
                    lon = points[i, j, k, 1]
                    alt = points[i, j, k, 2]
                    mask[i, j, k] = self.terrain.elevationAt(lat, lon) >= alt
        return mask
