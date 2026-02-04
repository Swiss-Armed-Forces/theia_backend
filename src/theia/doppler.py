import math
import numpy as np
import scipy.constants as sc

from theia.coordinates import get_azimuth_between_locs
from theia.distance import (
    burstvincentydistance,
    get_2d_distance_between_locs_heights,
    haversine,
)
from theia.types import Point, Radar, Target


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

    az = get_azimuth_between_locs(
        p_observer=Point(lat=rad_lat, lon=rad_lon, alt=rad_alt),
        p_target=Point(lat=tgt_lat, lon=tgt_lon, alt=tgt_alt),
    )  # radians
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


def calculate_bistatic_doppler(
    rx: Radar,
    tgt: Target,
    tx: Radar,
    dt: float = 1e-6,
):
    """Calculate bistatic Doppler in Hz.

    Parameters
    ----------
    rx: Radar
        Receiver.
    tgt: Target
        Target.
    tx: Radar
        Transmitter.
    dt: float
        Time step to use for the finite difference calculation. The target's
        velocity vector is assumed to be constant between the current time t
        and time t + dt.

    Returns
    -------
    float
        Doppler shift in [Hz]. This value can be negative, depending on the
        velocity vector of the target.

    Notes
    -----
    The bistatic Doppler shift is computed from the rate of change (R_t + R_r)
    divided by the wavelength of tx signal. Finite forward differences are used
    to calculate the Doppler shift.
    """

    rr1 = (
        get_2d_distance_between_locs_heights(
            tgt.lat,
            tgt.lon,
            tgt.alt,
            rx.lat,
            rx.lon,
            rx.alt + rx.antenna_height,
        )
        * 1000.0
    )
    rt1 = (
        get_2d_distance_between_locs_heights(
            tx.lat,
            tx.lon,
            tx.alt + tx.antenna_height,
            tgt.lat,
            tgt.lon,
            tgt.alt,
        )
        * 1000.0
    )

    tgt_total_vel = tgt.speed
    tgt_xy_vel = math.sqrt(tgt_total_vel * tgt_total_vel - tgt.vz * tgt.vz)  # [m/s]
    alpha = math.degrees(math.atan2(tgt.vlon, tgt.vlat))
    if alpha < 0:
        alpha = 360 + alpha  # now alpha is in degrees from north

    # Move the target along the bearing given by the velocity vectors with the
    # given velocity to calculate the change of the target position.
    new_lat_lon = burstvincentydistance((tgt.lat, tgt.lon), (tgt_xy_vel * dt), alpha)
    # This is the predicted target position with the given velocity.
    new_lat = new_lat_lon.latitude
    new_lon = new_lat_lon.longitude
    new_z = tgt.alt + dt * tgt.vz

    # Now compute bistatic range components R_T and R_R for the new target position.
    rr2 = (
        get_2d_distance_between_locs_heights(
            new_lat,
            new_lon,
            new_z,
            rx.lat,
            rx.lon,
            rx.alt + rx.antenna_height,
        )
        * 1000.0
    )
    rt2 = (
        get_2d_distance_between_locs_heights(
            tx.lat,
            tx.lon,
            tx.alt + tx.antenna_height,
            new_lat,
            new_lon,
            new_z,
        )
        * 1000.0
    )

    # Compute the rate of change for R_T (tx to target range) and R_R (tgt to rx range).
    rt_rate_of_change = (rt2 - rt1) / dt
    rr_rate_of_change = (rr2 - rr1) / dt

    wavelength = sc.speed_of_light / (tx.frequency * 1000000)  # m
    doppler_shift = (rt_rate_of_change + rr_rate_of_change) / wavelength  # [Hz]

    return doppler_shift
