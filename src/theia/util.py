import numpy as np


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
    return 10**(value / 10)
