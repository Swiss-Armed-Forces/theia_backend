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

def power_to_erp(power: float, losses: float, gain: float) -> float:
    """
    Calculate ERP [dBW].
    """
    return power + gain - losses - 2.15


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
