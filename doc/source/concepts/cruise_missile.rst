.. _cruise_missile:

Cruise missile
==============

A cruise missile (``CruiseMissileFactory``, listed under ``cruise_missiles`` in
an order of battle) flies from its launch point to its target at low altitude,
following the terrain. Its path is computed once, before the simulation starts,
by :func:`theia.cruise_missile.build_terrain_following_path`. In the simulation
it behaves like a one-way drone on a fixed path that engages its goal
once it is within 100 m of it.

Inputs
------

.. list-table::
   :header-rows: 1

   * - Field
     - Unit
     - Default
     - Meaning
   * - ``p_start``, ``p_stop``
     - °
     - (required)
     - Launch and target point. The starting point altitude is increased
       in case no trajectory that starts at that altitude can be found.
   * - ``t_start``
     - datetime
     - (required)
     - Launch time.
   * - ``speed``
     - m/s
     - (required)
     - Constant speed along the 3D path, including the dive.
   * - ``cruise_magl``
     - m
     - (required)
     - Nominal cruise height above ground level (AGL).
   * - ``min_flight_path_angle``
     - °
     - −10
     - Steepest descent while cruising, in (−90°, 0°).
   * - ``max_flight_path_angle``
     - °
     - 15
     - Steepest climb while cruising, in (0°, 90°).
   * - ``terminal_dive_angle``
     - °
     - −30
     - Flight-path angle of the terminal dive, in (−90°, 0°).
   * - ``min_clearance``
     - m
     - min(``cruise_magl``, max(15 m, 0.5 · ``cruise_magl``))
     - Hard floor above the terrain, in (0, ``cruise_magl``].
   * - ``sample_spacing``
     - m
     - 100
     - Distance between trajectory samples along the ground track.
   * - ``terrain``, ``target_id``, ``effector_id``, ``rcs``, ``category``
     -
     - ``category``: ``CRUISE_MISSILE``
     - As for ballistic missiles.

The defaults and the algorithm's tuning constants are defined in
:mod:`theia.config` with the prefix ``CM_`` (e.g.
``CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE``).

Altitudes are metres above sea level, like everywhere else in Theia. Heights
above ground are measured against the terrain model's ``elevationAt``.

Flight path
-----------

**Ground track.** The missile follows the great circle from launch to target
on a spherical earth (the same model as the ``haversine`` distance).

**Cruise profile.** The terrain is sampled along the track at 30 m or finer.
The cruise profile is the *lowest* profile that

- never goes below ``terrain + cruise_magl``, and
- keeps every flight-path angle within
  [``min_flight_path_angle``, ``max_flight_path_angle``].

It is computed in two linear passes. A backward pass starts climbs early enough
to clear upcoming ridges at the maximum climb angle. A forward pass then limits
descents to the minimum flight-path angle, so the missile stays high over
valleys too narrow to descend into.

**Launch.** The missile starts at ``terrain(p_start) + cruise_magl``. If the
terrain right after the launch rises faster than the missile can climb, it is
launched correspondingly higher.

**Terminal dive.** The missile dives in a straight line at
``terminal_dive_angle`` onto ``(p_stop.lat, p_stop.lon, terrain(p_stop))``.
The dive starts where this line meets the cruise profile.

**Timing.** Time is the 3D path length along the straight segments between the
trajectory samples, divided by ``speed``, starting at ``t_start``. Velocities are not modelled (they are 0 in the trajectory); derive
them from the positions if needed.

**Impact.** The impact point is repeated for 1 s after the impact
(``CM_IMPACT_HOLD_OFFSETS``). This guarantees that a 1 s simulation time step
catches the missile within its combat range of ``CM_COMBAT_RANGE`` meters. The
hold forces the cubic spline to stop abruptly. Closely spaced samples on the dive
line shortly before the impact (``CM_PRE_IMPACT_OFFSETS``) and after it confine
that distortion to a few hundredths of a second: the interpolated position stays
within a few metres of the planned path and of the impact point.

Interpolation
-------------

The trajectory is sampled every ``sample_spacing`` metres, plus the dive start,
the pre-impact samples and the impact point. Regular samples closer than
``CM_MIN_KNOT_SPACING_FACTOR * sample_spacing`` to these are dropped, because
near-duplicate spline knots cause overshoot. The simulator interpolates between samples with a cubic
spline (see ``Trajectory``). This has two consequences:

- The interpolated trajectory is checked against the terrain: while cruising it
  must stay at least ``min_clearance`` above the terrain, and during the dive it
  must not go below the terrain.
- The flight-path angle limits hold at the samples. Where the profile switches
  abruptly between climbing, following the terrain and descending (ridge tops,
  valley bottoms), the spline rounds the corner and may locally exceed the
  limits by a few degrees. Smaller sample spacings do not remove this effect.

Errors
------

``ValueError`` is raised (``pydantic.ValidationError`` when loading a file) if

- a parameter is outside its valid range,
- launch and target are less than ``2 * sample_spacing`` apart,
- the target is too close for the terminal dive,
- the terminal dive is blocked by terrain, or
- the interpolated trajectory comes closer to the terrain than allowed (reduce
  ``sample_spacing``).

The parameters are validated when the order of battle is loaded. The path
itself, which needs the terrain, is computed when the simulation is set up.
