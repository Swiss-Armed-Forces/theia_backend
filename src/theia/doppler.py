import numpy as np
import scipy.constants as sc

from theia.coordinates import CoordinateTransformations
from theia.types import Receiver, Target, Transmitter


def calculate_doppler_shift(
    rx: Receiver,
    tgt: Target,
    tx: Transmitter,
    dt: float = 1e-6,
):
    """
    Calculate Doppler shift [Hz] assuming transmitter and receiver
    do not move.

    Parameters
    ----------
    rx: Receiver
    tgt: Target
    tx: Transmitter
    dt: float
        Time step to use for the finite difference calculation. The target's
        velocity vector is assumed to be constant between the current time t
        and time t + dt.

    Returns
    -------
    float
        Doppler shift in [Hz]
        Negative value: The bistatic range is getting smaller ("target is approaching").
        Positive value: The bistatic range is getting smaller ("target is leaving).

    Notes
    -----
    The bistatic Doppler shift is computed from the rate of change (R_t + R_r)
    divided by the wavelength of tx signal. Finite forward differences are used
    to calculate the Doppler shift.
    """
    # Convert positions to Cartesian space.
    rx_xyz = np.asarray(
        CoordinateTransformations.geodetic_to_cartesian(
            rx.lat, rx.lon, rx.alt + rx.antenna_height
        )
    )
    tx_xyz = np.asarray(
        CoordinateTransformations.geodetic_to_cartesian(
            tx.lat, tx.lon, tx.alt + tx.antenna_height
        )
    )
    target_xyz = np.asarray(
        CoordinateTransformations.geodetic_to_cartesian(*tgt.point.as_tuple())
    )

    # Calculate target position at time + dt in the future.
    target_xyz_future = target_xyz + np.asarray(tgt.velocity.as_tuple()) * dt

    # Calculate difference in signal path length now and at time + dt in the future.
    signal_path = np.linalg.norm(target_xyz - rx_xyz) + np.linalg.norm(
        target_xyz - tx_xyz
    )
    signal_path_future = np.linalg.norm(target_xyz_future - rx_xyz) + np.linalg.norm(
        target_xyz_future - tx_xyz
    )

    diff = signal_path_future - signal_path

    wavelength = sc.speed_of_light / (tx.frequency * 1000000)  # m
    doppler_shift = diff / wavelength  # [Hz]

    return doppler_shift
