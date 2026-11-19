import datetime
import os
import pathlib

ELEVATION_DATA_DIR = os.environ.get("THEIA_ELEVATION_DATA_DIR")
TERRAIN_HBV_DATA_DIR = os.environ.get(
    "THEIA_HBV_TERRAIN_DATA_DIR",
    pathlib.Path(__file__).parent.parent.parent,  # repo root
)
DEFAULT_TERRAIN = os.environ.get("THEIA_DEFAULT_TERRAIN", "SRTM")

ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD = 40.0
"""
Doppler shift threshold for active radar [Hz].

Targets with less Doppler shift are not detected. This models the clutter
filter (MTI) of a ground-based radar: Static targets (e.g. radar sites) are not
detected. The threshold corresponds to a minimum radial speed of
``threshold * wavelength / 2``, i.e. 2 m/s at 3 GHz and 1 m/s at 6 GHz.
A one-way drone at 200 km/h is therefore only missed within ~2° of tangential
flight.

A negative value disables the threshold. The Doppler shift calculation assumes
a stationary radar, so the threshold is not suitable for moving (e.g.
aircraft-mounted) radars.
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

BM_COMBAT_RANGE = 100.0
"""Ballistic missile: distance to its goal within which the missile attacks (detonates) [m]"""

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
CM_DEFAULT_CRUISE_MAGL = 50.0
"""
Cruise missile: default nominal cruise height above ground level [m] of a
round launched in the interactive game without an explicit path generator.
"""

DRONE_DEFAULT_CRUISE_ALT_MASL = 5000.0
"""One-way drone: default cruise altitude of the climb-cruise-dive path [m above sea level]"""
DRONE_DEFAULT_DIVE_START_DISTANCE = 1000.0
"""
One-way drone: default distance along the ground track before the target at
which the terminal dive starts [m]
"""
DRONE_DEFAULT_CLIMB_ANGLE = 10.0
"""One-way drone: default flight-path angle of the initial climb [°]"""
DRONE_DEFAULT_COMBAT_RANGE = 100.0
"""One-way drone: distance to its goal within which the drone attacks (detonates) [m]"""
DRONE_SAMPLE_SPACING = 100.0
"""
One-way drone: maximum distance between trajectory samples along the ground
track of the climb-cruise-dive path [m]
"""

INTEL_DETECTION_SIGMA = 0.1
"""
Intel sensing: standard deviation [m] of a position sampled from an intel detection.

Must be > 0 so that the Kalman covariance stays regular.
"""

PT_UNKNOWN_POSITION_VARIANCE = 1e12
"""
Pseudo-tracker: variance [m^2] of each ECEF position component of a track's
prior, i.e. before its first detection.

Large compared to any measurement variance so that the first detection alone
determines the initial position estimate (1e12 m^2 = (1000 km)^2).
"""

FRONTEND_URL = "http://localhost:5173"


UNKNOWN_ID = -1
UNKNOWN_TIME = datetime.datetime.fromtimestamp(0, datetime.UTC)
UNKNOWN_RCS = -1.0
"""
Radar cross section [m^2] of a target whose RCS is unknown, e.g. a target
revealed by intel.

Negative, so it cannot be mistaken for a physical value.
"""
