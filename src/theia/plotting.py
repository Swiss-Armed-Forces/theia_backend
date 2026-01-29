from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import shapely
from theia.coordinates import CoordinateTransformations
from theia.data_loading import elevationAt
from theia.distance import line_of_sight_distance, linspace
from theia.ellipsoid import Ellipsoid
from theia.types import Point


def plot_profile(
    p1: Point,
    p2: Point,
    p1_label: str = "",
    p2_label: str = "",
    plot_connection_p1_p2: bool = True,
    resolution: float = 30.0,
    min_elevation: float = 0.,
) -> tuple[plt.Figure, plt.Axes]:
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    points = list(linspace(p1, p2, resolution))
    distances = [
        line_of_sight_distance(p1.lat, p1.lon, p1.alt, p.lat, p.lon, p.alt) / 1000.
        for p in points
    ]
    elevations = np.clip([elevationAt(p.lat, p.lon) for p in points], 0, np.inf)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(distances, elevations, label="Terrain")
    ax.plot([distances[0]], [p1.alt], "o")
    ax.plot([distances[-1]], [p2.alt], "o")
    ax.annotate(
        p1_label,
        (distances[0], p1.alt),
        xytext=(2 * resolution / 1000., p1.alt + 2 * resolution),
        color=colors[1],
        fontweight="bold",
    )
    ax.annotate(
        p2_label,
        (distances[-1], p2.alt),
        xytext=(distances[-1] - 4 * resolution / 1000., p2.alt),
        horizontalalignment="right",
        color=colors[2],
        fontweight="bold",
    )
    if plot_connection_p1_p2:
        ax.plot([distances[0], distances[-1]], [p1.alt, p2.alt], "--")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Altitude (MASL) [m]", size=16)
    ax.set_xlabel("Geodetic Distance [km]", size=16)
    ax.grid(True)

    return fig, ax


def detection_ellipse(
    Tx_position: Point,
    Rx_position: Point,
    bistatic_range: float,
    target_altititude: float,
    altitude_slice_thickness: float = 100,
) -> shapely.geometry.LineString:
    ellipsoid = Ellipsoid(
        CoordinateTransformations.geodetic_to_cartesian(*Tx_position.as_tuple()),
        CoordinateTransformations.geodetic_to_cartesian(*Rx_position.as_tuple()),
        bistatic_range
        + line_of_sight_distance(
            Tx_position.lat,
            Tx_position.lon,
            Tx_position.alt,
            Rx_position.lat,
            Rx_position.lon,
            Rx_position.alt,
        ),
    )

    points = ellipsoid.sample_surface(n_phi=360, n_theta=360)
    points = [CoordinateTransformations.cartesian_to_geodetic(*p) for p in points]
    points_at_target_alt = [
        p[:2]
        for p in points
        if abs(p[2] - target_altititude) < 0.5 * altitude_slice_thickness
    ]

    hull = ConvexHull(points_at_target_alt)
    points_at_target_alt = hull.points[hull.vertices]
    ellipse = shapely.geometry.LineString(
        [(p[1], p[0]) for p in points_at_target_alt]
    )
    return ellipse