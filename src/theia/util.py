from collections import deque
import functools
import logging
import time
import uuid

import folium
from folium.map import CustomPane
import numpy as np
from scipy import integrate, special
import scipy.constants as sc
import shapely
from skimage import measure


def erp_to_power(erp: float, losses: float, gain: float) -> float:
    """
    Convert Effective Radiated Power (ERP) to power.

    The formula from `[1]`_.

    .. _[1]: https://en.wikipedia.org/wiki/Effective_radiated_power

    Parameters
    ----------
    erp: float
        ERP [dBW]
    losses: float
        Losses [dB]
    gain: float
        Gain [dBi]

    Returns
    -------
    power: float
        Output power of the transmitter [W]
    """
    # Add 2.15dBW so we use EIRP in the radar equation.
    power_dBW = erp + 2.15 + losses - gain
    return from_dB(power_dBW)


def to_dB(value: float) -> float:
    return 10 * np.log10(value)


def from_dB(value: float) -> float:
    return 10 ** (value / 10)


def marcum_q_integrand(v: float, alpha: float) -> float:
    return v * np.exp(-(v * v + alpha * alpha) / 2) * special.iv(0, alpha * v)


def marcum_q_function(alpha: float, beta: float) -> float:
    """
    Calculate the values of the marcum q function.

    Parameters
    ----------
    alpha: float
        alpha parameter; must be <= 30
    beta: float
        beta parameter

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If alpha is > 30 due to numerical instability of the Bessel function
    """
    if alpha > 30.0:
        raise ValueError("alpha must be <= 30 to avoid overflow in Bessel functions!")
    return (
        1
        - integrate.quad(
            lambda x: marcum_q_integrand(x, alpha),
            0,
            beta,
        )[0]
    )


def get_clear_sky_attenuation(transmitter_freq: float) -> float:
    """
    Calculate clear sky atmospheric one-way attenuation [dB/km] for Radar Windows.

    Parameters
    ----------
    transmitter_freq: float
        Transmitter frequency [MHz].

    Returns
    -------
    float
        Clear sky atmospheric one-way attenuation [dB/km]

    Notes
    -----
    Clear Sky weather values from Barton book: 'Modern Radar System Analysis'.
    """

    freq_mhz = [200, 500, 1000, 10000]
    atten = [0.00075, 0.003, 0.0055, 0.012]
    # one-way attenuation: dB/km therefore division by 2
    # ???
    atten_db = max(np.interp(transmitter_freq, freq_mhz, atten), 0)

    return atten_db


def frequency_to_wavelength(frequency: float) -> float:
    """
    Calculate wavelength for given frequency.

    Parameters
    ----------
    frequency:
        Frequency [MHz]

    Returns
    -------
    wavelength: float
        Wavelength [m]
    """
    return sc.speed_of_light / (frequency * 1e6)


def mask_to_polygon(
    mask: np.ndarray,
    x0: float,
    dx: float,
    y0: float,
    dy: float,
) -> list[shapely.Polygon]:
    def index_to_lonlat(ix: int, iy: int) -> tuple[float, float]:
        return (y0 + iy * dy, x0 + ix * dx)

    def signed_area(points):
        return 0.5 * sum(
            points[i, 0] * points[(i + 1) % len(points), 1]
            - points[(i + 1) % len(points), 0] * points[i, 1]
            for i in range(len(points))
        )

    mask = np.pad(
        mask.astype(np.uint8), pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(mask, level=0.5)

    outers = []
    holes = []
    for contour in contours:
        points = np.array([index_to_lonlat(p[0], p[1]) for p in contour])
        if signed_area(points) > 0:
            outers.append(points)
        else:
            holes.append(points)

    final_polygons = []
    for outer in outers:
        outer_poly = shapely.Polygon(outer)
        inner_holes = []

        for hole in holes:
            if outer_poly.contains(shapely.Point(hole[0])):
                inner_holes.append(hole)

        final_polygons.append(shapely.Polygon(outer, holes=inner_holes))
    return final_polygons


def normal_pdf(mean: float, sigma: float, x: float) -> float:
    if sigma > 0:
        prob = (
            1 / np.sqrt(2 * np.pi * sigma) * np.exp(-((mean - x) ** 2) / (2 * sigma**2))
        )
    elif np.isclose(x, mean):
        prob = 1.0
    else:
        prob = 0.0

    return prob


class TimingStats:
    def __init__(self, name, window=200):
        self.name = name
        self.samples = deque(maxlen=window)
        self.call_count = 0

    def record(self, elapsed):
        self.samples.append(elapsed)
        self.call_count += 1
        if self.call_count % 10 == 0:
            s = sorted(self.samples)
            n = len(s)
            logging.warning(
                f"[{self.name}] n={n} "
                f"p50={s[n // 2] * 1000:.2f}ms "
                f"p90={s[int(n * 0.9)] * 1000:.2f}ms "
                f"max={s[-1] * 1000:.2f}ms",
            )


def timed(name):
    stats = TimingStats(name)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            stats.record(time.perf_counter() - t0)
            return result

        return wrapper

    return decorator


def add_blurred_tile_layer(
    m: folium.Map,
    tiles: str,
    attr: str = None,
    blur_amount: float = 8,
    grayscale: bool = False,
    z_index: int = 200,
    name: str = None,
    **tile_layer_kwargs,
) -> folium.TileLayer:
    """
    Add a TileLayer to a folium Map whose tiles are visually blurred via CSS.

    Useful for scrambling plots for conferences.

    Parameters
    ----------
    m : folium.Map
        The map to add the layer to.
    tiles : str
        Tile URL template or named tileset, same as folium.TileLayer(tiles=...).
    attr : str, optional
        Attribution string (Leaflet requires this for custom tile URLs).
    blur_amount : float
        Blur radius in pixels. Default 8.
    grayscale : bool
        Also desaturate the tiles. Default False.
    z_index : int
        Pane z-index. Default 200 (Leaflet's default tile-pane tier).
    name : str, optional
        Layer name shown in LayerControl.
    **tile_layer_kwargs
        Passed through to folium.TileLayer (e.g. opacity, overlay, show).

    Returns
    -------
    folium.TileLayer
        The tile layer that was added (already attached to `m`).
    """
    pane_id = uuid.uuid4().hex[:8]
    pane_name = f"blurred{pane_id}"

    # Create the dedicated pane.
    pane = CustomPane(pane_name, z_index=z_index, pointer_events=False)
    pane.add_to(m)

    # Apply the blur via CSS. Leaflet's createPane gives the pane div the
    # class `leaflet-<name>-pane`, so we target that class directly.
    filters = [f"blur({blur_amount}px)"]
    if grayscale:
        filters.append("grayscale(1)")
    css = f"""
    <style>
        .leaflet-{pane_name}-pane {{
            filter: {" ".join(filters)};
        }}
    </style>
    """
    m.get_root().header.add_child(folium.Element(css))

    # Attach the tile layer to that pane so only these tiles are blurred.
    layer = folium.TileLayer(
        tiles=tiles,
        attr=attr,
        name=name,
        pane=pane_name,
        **tile_layer_kwargs,
    )
    layer.add_to(m)
    return layer


def angle_in_interval(
    angle: float | np.typing.ArrayLike,
    low: float | np.typing.ArrayLike,
    high: float | np.typing.ArrayLike,
    is_rad: bool = False,
) -> bool | np.typing.NDArray[np.bool_]:
    """
    Check whether `angle` lies within [low, high], handling wraparound.
 
    Parameters
    ----------
    angle : float or array-like
        Angle(s) to test.
    low : float
        Lower bound of the interval.
    high : float
        Upper bound of the interval. May be numerically smaller than
        `low` to express an interval that wraps around (e.g. [270, 90]).
    is_rad : bool, default False
        If True, angles/bounds are interpreted in radians (period 2*pi).
        If False (default), they are interpreted in degrees (period 360).
 
    Returns
    -------
    bool or numpy.ndarray of bool
        Whether each angle lies within [low, high].
 
    Notes
    -----
    Special case: if `high - low` is a nonzero multiple of the period
    (e.g. low=0, high=360), the interval is treated as a full circle and
    matches every angle. A genuine zero-width interval (low == high
    exactly) still matches only that single angle.
    """
    period = 2 * np.pi if is_rad else 360
 
    span = high - low
 
    # Full-circle case: distinguish from a zero-width point by looking at
    # the *raw* span before reducing it mod period. [0, 360] has span=360
    # (full circle); [30, 30] has span=0 (single point).
    if span != 0 and span % period == 0:
        if np.ndim(angle) > 0:
            return np.ones(np.shape(angle), dtype=bool)
        return True
 
    width = span % period
    offset = (np.asarray(angle) - low) % period
    result = offset <= width
    return result if np.ndim(angle) > 0 else bool(result)
