import os


ELEVATION_DATA_DIR = os.environ.get("THEIA_ELEVATION_DATA_DIR")
ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD = 5.0
"""
Doppler shift threshold for active radar [Hz].

Targets with less Doppler shift are not detected.
"""
NOISE_TEMPERATURE = 300
"""Noise temperature [K]"""