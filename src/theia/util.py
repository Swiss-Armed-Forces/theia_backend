import numpy as np
from scipy import integrate, special


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
    power_dBW = erp + losses - gain
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