import numpy as np
import scipy.constants as sc

from theia.coordinates import get_azimuth_between_locs
from theia.distance import haversine


def monostatic_doppler(
    freq: float,
    rad_lat: float,
    rad_lon: float,
    rad_alt: float,
    tgt_lat: float,
    tgt_lon: float,
    tgt_alt: float,
    tgt_vx: float,
    tgt_vy: float,
    tgt_vz: float,
):
    """computes Doppler shift in Hz from target motion as seen at ground stationary radar
    freq given in MHz, alts given in masl.
    returns None if tgt V == 0. tgt_V is given as m/s along the lon/lat/z axis
    see Class Target definition

    Parameters
    ----------
    freq: float
        Radar frequency [MHz]
    rad_lat: float
        Radar latitude [°]
    rad_lon: float
        Radar longitude [°]
    rad_alt: float
        Radar altitude above sea level [m]
    tgt_lat: float
        Target latitude [°]
    tgt_lon: float
        Target longitude [°]
    tgt_alt: float
        Target altitude above sea level [m]
    tgt_vx: float
        Target velocity along longitude axis [m / s]
    tgt_vy: float
        Target velocity along latitude axis [m / s]
    tgt_vz: float
        Target velocity along radial axis [m / s]

    Returns
    -------
    doppler_shift: float
        Doppler shift from target motion [Hz]; np.nan if the velocity is zero
    """
    if (tgt_vx == 0) and (tgt_vy == 0) and (tgt_vz == 0):
        return np.nan

    az = get_azimuth_between_locs(rad_lat, rad_lon, tgt_lat, tgt_lon)  # radians
    xy_dist = haversine(rad_lat, rad_lon, tgt_lat, tgt_lon)
    # (gx,gy,gz) will be the vector looking at the target from the radar
    gx = xy_dist * np.cos(az)  # [m]
    gy = xy_dist * np.sin(az)  # [m]
    gz = tgt_alt - rad_alt  # [m]

    # now find the aspect angle (ie angle between (tgt_vx, tgt_vy, tgt_vz) and (gx,gy,gz))
    g = np.array([gx, gy, gz])  # [m]
    tgt_v = np.array([tgt_vx, tgt_vy, tgt_vz])  # [m/s]

    # project tgt_v on g
    proj_tgt_v = project_vector_u_on_v(tgt_v, g)

    # calculate aspect angle
    asp_angle = calc_angle_from_vecs(g, tgt_v)

    # get wavelength from freq where freq is given in GHz
    wv = sc.c / (freq * 1000000)  # [m]

    # now compute Doppler shift
    dopp_shift = 2.0 * np.linalg.norm(proj_tgt_v) * np.cos(asp_angle) / wv
    return dopp_shift


def project_vector_u_on_v(u, v):
    """returns projection of vector u on vector v, u and v given as numpy arrays"""
    # finding norm of the vector v
    v_norm = np.sqrt(sum(v**2))
    proj_of_u_on_v = (np.dot(u, v) / v_norm**2) * v
    return proj_of_u_on_v


def calc_angle_from_vecs(v1, v2):
    """returns the inner angle between two vectors"""
    return np.atan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
