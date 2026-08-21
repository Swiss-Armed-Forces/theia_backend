import math

import scipy.constants as sc

from theia.coordinates import CoordinateTransformations
from theia.types import Receiver, Target, Transmitter


def calculate_doppler_shift(
    rx: Receiver,
    tgt: Target,
    tx: Transmitter,
    dt: float = 1e-6,
):
    r"""
    Calculate Doppler shift [Hz] assuming transmitter and receiver
    do not move.

    Parameters
    ----------
    rx: Receiver
        Receiver
    tgt: Target
        Target
    tx: Transmitter
        Transmitter
    dt: float
        Time step [s] to use for the finite difference calculation. The target's
        velocity vector is assumed to be constant between the current time t
        and time t + dt.

    Returns
    -------
    float
        Doppler shift in [Hz]

        * Negative value: The bistatic range is getting bigger ("target is leaving").
        * Positive value: The bistatic range is getting smaller ("target is approaching").

    Notes
    -----
    The bistatic Doppler shift is computed from the rate of change (R_t + R_r)
    divided by the wavelength of tx signal. Finite forward differences are used
    to calculate the Doppler shift.
    """
    # Convert positions to Cartesian space.
    rx_xyz = CoordinateTransformations.geodetic_to_cartesian(
        rx.lat,
        rx.lon,
        rx.alt + rx.antenna_height,
    )
    tx_xyz = CoordinateTransformations.geodetic_to_cartesian(
        tx.lat,
        tx.lon,
        tx.alt + tx.antenna_height,
    )
    target_xyz = CoordinateTransformations.geodetic_to_cartesian(
        tgt.point.lat,
        tgt.point.lon,
        tgt.point.alt,
    )

    # Calculate target position at time + dt in the future.
    target_xyz_future = (
        target_xyz[0] + tgt.velocity.vx * dt,
        target_xyz[1] + tgt.velocity.vy * dt,
        target_xyz[2] + tgt.velocity.vz * dt,
    )

    # Calculate difference in signal path length now and at time + dt in the future.
    d_rx_target = math.sqrt(
        (target_xyz[0] - rx_xyz[0]) ** 2
        + (target_xyz[1] - rx_xyz[1]) ** 2
        + (target_xyz[2] - rx_xyz[2]) ** 2
    )
    d_tx_target = math.sqrt(
        (target_xyz[0] - tx_xyz[0]) ** 2
        + (target_xyz[1] - tx_xyz[1]) ** 2
        + (target_xyz[2] - tx_xyz[2]) ** 2
    )
    d_rx_target_future = math.sqrt(
        (target_xyz_future[0] - rx_xyz[0]) ** 2
        + (target_xyz_future[1] - rx_xyz[1]) ** 2
        + (target_xyz_future[2] - rx_xyz[2]) ** 2
    )
    d_tx_target_future = math.sqrt(
        (target_xyz_future[0] - tx_xyz[0]) ** 2
        + (target_xyz_future[1] - tx_xyz[1]) ** 2
        + (target_xyz_future[2] - tx_xyz[2]) ** 2
    )
    signal_path = d_rx_target + d_tx_target
    signal_path_future = d_rx_target_future + d_tx_target_future

    diff = signal_path_future - signal_path

    wavelength = sc.speed_of_light / (tx.frequency * 1000000)  # m
    doppler_shift = diff / (wavelength * dt)

    return doppler_shift
