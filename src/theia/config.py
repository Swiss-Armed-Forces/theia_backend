import datetime
import enum
import os


ELEVATION_DATA_DIR = os.environ.get("THEIA_ELEVATION_DATA_DIR")
# Negative value means no Doppler thresholding, which is necessary for aircraft-mounted RAD
ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD = -1
"""
Doppler shift threshold for active radar [Hz].

Targets with less Doppler shift are not detected.
"""
NOISE_TEMPERATURE = 300
"""Noise temperature [K]"""
RF_LOSS = 12.0
"""RF system hardware loss [dB]"""
SNR_THRESHOLD_PCL = 15.0
"""Signal-to-noise ratio threshold for passive coherent location [dB]"""
DOPPLER_SHIFT_THRESHOLD_PCL = 2.0
"""
Doppler threshold [Hz] for PCL.

Targets with less Doppler shift will not be detected.
"""
DELAY_THRESHOLD_PCL = 1.0
"""
Delay threshold for PCL [us].

This is used to judge whether a given transmitter - target - receiver geometry
is in the bistatic or the forward scattering regime.
"""

RCS_FOR_RANGE_CALCULATION = 1.0
"""Radar cross section [m^2] to be used for monostatic range calculations by default."""

FRONTEND_URL = "http://localhost:5173"


class SIDC(enum.Enum):
    GREEN_TRANSMITTER = "10242000001212010000"
    BLUE_RECEIVER = "10231500002203000000"
    RED_RECEIVER = "10261500002203000000"
    BLUE_RADAR = "10231500002203000000"
    RED_RADAR = "10261500002203000000"
    RED_FIXED_WING = "10260100001101000000"
    BLUE_AIR_DEFENCE = "10031000001301000000"
    RED_AIR_DEFENCE = "10061000001301000000"
    BLUE_GOVERNMENT_SITE = "10232000001206000000"
    BLUE_AIRPORT = "10032000001213010000"
    RED_MISSILE = "10260200001100000000"
    UNKNOWN = "10211000000000000000"


UNKNOWN_ID = -1
UNKNOWN_TIME = datetime.datetime.fromtimestamp(0)
