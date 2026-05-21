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

SIDC_GREEN_TRANSMITTER = "10242000001212010000"
SIDC_BLUE_RECEIVER = "10231500002203000000"
SIDC_RED_RECEIVER = "10261500002203000000"
SIDC_BLUE_RADAR = "10231500002203000000"
SIDC_RED_RADAR = "10261500002203000000"
SIDC_RED_FIXED_WING = "10260100001101000000"
SIDC_UNKNOWN = "10211000000000000000"