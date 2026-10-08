import datetime
import os
import pathlib

ELEVATION_DATA_DIR = os.environ.get("THEIA_ELEVATION_DATA_DIR")
TERRAIN_HBV_DATA_DIR = os.environ.get(
    "THEIA_HBV_TERRAIN_DATA_DIR",
    pathlib.Path(__file__).parent.parent.parent,  # repo root
)
DEFAULT_TERRAIN = os.environ.get("THEIA_DEFAULT_TERRAIN", "SRTM")

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

CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE = -10.0
"""Cruise missile: default steepest descent while cruising [°]"""
CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE = 15.0
"""Cruise missile: default steepest climb while cruising [°]"""
CM_DEFAULT_TERMINAL_DIVE_ANGLE = -30.0
"""Cruise missile: default flight-path angle of the terminal dive [°] (negative = descending)"""
CM_DEFAULT_SAMPLE_SPACING = 100.0
"""Cruise missile: default distance between trajectory samples along the ground track [m]"""
CM_DEFAULT_MIN_CLEARANCE_FLOOR = 15.0
"""Cruise missile: lower bound of the default minimum clearance above the terrain [m]"""
CM_DEFAULT_MIN_CLEARANCE_FACTOR = 0.5
"""
Cruise missile: default minimum clearance as a fraction of the cruise height
above ground level [-].

The default minimum clearance is
``min(cruise_magl, max(CM_DEFAULT_MIN_CLEARANCE_FLOOR, CM_DEFAULT_MIN_CLEARANCE_FACTOR * cruise_magl))``.
"""
CM_COMBAT_RANGE = 100.0
"""Cruise missile: distance to its goal within which the missile attacks (detonates) [m]"""
CM_TERRAIN_SAMPLE_SPACING = 30.0
"""Cruise missile: maximum spacing at which the terrain is sampled along the track [m]"""
CM_IMPACT_HOLD_OFFSETS = (0.05, 0.2, 0.5, 1.0)
"""
Cruise missile: times after impact [s] at which the impact point is repeated.

The simulator only evaluates the trajectory at discrete time steps. Holding the
impact point for a second guarantees that a 1 s time step catches the missile
within its combat range. Several closely spaced samples keep the cubic spline
from overshooting far past (and below) the impact point.
"""
CM_PRE_IMPACT_OFFSETS = (0.05, 0.1, 0.2, 0.3, 0.5)
"""
Cruise missile: times before impact [s] at which extra samples are placed on the
dive line.

The impact hold (see ``CM_IMPACT_HOLD_OFFSETS``) forces the cubic spline to slow
down abruptly. Closely spaced samples before the impact confine that distortion
to the last fraction of a second instead of the whole last sample interval.
"""
CM_MIN_KNOT_SPACING_FACTOR = 0.25
"""
Cruise missile: regular trajectory samples closer than this fraction of
``sample_spacing`` to the dive start, the pre-impact samples or the impact are
dropped [-] because near-duplicate knots next to much longer intervals make
the cubic spline overshoot.
"""
CM_DIVE_PENETRATION_TOLERANCE = 0.5
"""Cruise missile: tolerated interpolation dip below the terrain during the terminal dive [m]"""

FRONTEND_URL = "http://localhost:5173"


UNKNOWN_ID = -1
UNKNOWN_TIME = datetime.datetime.fromtimestamp(0, datetime.UTC)
