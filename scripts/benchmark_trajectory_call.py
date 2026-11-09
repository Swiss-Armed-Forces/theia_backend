"""
Quick benchmark harness for theia.types.Trajectory.__call__.

Usage
-----
    python scripts/benchmark_trajectory_call.py
    python scripts/benchmark_trajectory_call.py --profile trajectory_call

Design
------
See benchmark_coordinate_transformations.py for the general approach (fixed
randomly-sampled batch, timeit.repeat best-of-N, warm-up call).

Trajectory.__call__ evaluates the cubic spline and wraps the result in a fully
validated Target (plus Point and Velocity). In the mission_simulation notebook
profile it was called ~4.4M times, ~3.1M of them from
trajectory_intersects_terrain, which only needs lat/lon/alt. The benchmarks
split the per-call cost into its parts so an optimization can be judged
against its lower bound:
  - trajectory_call:          the full call, as used by the simulator
  - spline_scalar:            Trajectory's own spline evaluation only
  - spline_scipy_scalar:      the same with scipy's CubicSpline, one time per call
  - spline_scipy_vectorized:  all times in one CubicSpline call (lower bound for
                              batch callers like trajectory_intersects_terrain)
  - target_validated:         Target(...) with prebuilt Point/Velocity
  - target_model_construct:   Target.model_construct(...), skipping validation
  - intersects_terrain:       trajectory_intersects_terrain on a flat dummy
                              terrain that is never hit, so every segment is
                              fully sampled (per-call figure is per trajectory)
"""

import argparse
import cProfile
import datetime
import pstats
from typing import Callable

import numpy as np
import timeit
from scipy.interpolate import CubicSpline

from theia.terrain import DummyTerrain, trajectory_intersects_terrain
from theia.types import (
    ConstantRcsModel,
    Party,
    Point,
    Target,
    TargetInfos,
    Trajectory,
    Velocity,
)

SEED = 20261030
BATCH_SIZE = 2_000
TIMEIT_REPEAT = 7
N_WAYPOINTS = 20
"""Cruise-missile-like path: ~20 waypoints, ~10 km apart, ~250 m/s."""
N_TERRAIN_CHECKS = 5
"""trajectory_intersects_terrain is far slower per call, so use a smaller batch."""


def _sample_trajectory(rng: np.random.Generator) -> Trajectory:
    t0 = datetime.datetime(year=2026, month=10, day=30)
    times = [t0 + datetime.timedelta(seconds=40.0 * i) for i in range(N_WAYPOINTS)]
    lats = (47.0 + np.cumsum(rng.uniform(0.0, 0.09, size=N_WAYPOINTS))).tolist()
    lons = (8.0 + np.cumsum(rng.uniform(-0.05, 0.05, size=N_WAYPOINTS))).tolist()
    alts = rng.uniform(100.0, 1_000.0, size=N_WAYPOINTS).tolist()
    return Trajectory(
        target_id=0,
        target_info=TargetInfos.FIXED_WING.with_party(Party.RED),
        times=times,
        lats=lats,
        lons=lons,
        alts=alts,
        vxs=[0.0] * N_WAYPOINTS,
        vys=[0.0] * N_WAYPOINTS,
        vzs=[0.0] * N_WAYPOINTS,
        cross_section_model=ConstantRcsModel(rcs=0.1),
    )


def _sample_times(
    trajectory: Trajectory, n: int, rng: np.random.Generator
) -> list[datetime.datetime]:
    duration = (trajectory.times[-1] - trajectory.times[0]).total_seconds()
    offsets = np.sort(rng.uniform(0.0, duration, size=n))
    return [trajectory.times[0] + datetime.timedelta(seconds=s) for s in offsets]


def build_benchmarks() -> dict[str, tuple[Callable[[], None], int]]:
    """Map benchmark name to (batch call, number of calls per batch)."""
    rng = np.random.Generator(np.random.PCG64(seed=SEED))
    trajectory = _sample_trajectory(rng)
    times = _sample_times(trajectory, BATCH_SIZE, rng)
    timestamps = [t.timestamp() for t in times]
    timestamps_array = np.array(timestamps)
    spline = trajectory._spline
    scipy_spline = CubicSpline(
        [t.timestamp() for t in trajectory.times],
        np.stack(
            [
                trajectory.lats,
                trajectory.lons,
                trajectory.alts,
                trajectory.vxs,
                trajectory.vys,
                trajectory.vzs,
            ],
            axis=1,
        ),
        extrapolate=True,
    )

    point = Point(lat=47.5, lon=8.5, alt=500.0)
    velocity = Velocity(vx=0.0, vy=0.0, vz=0.0)
    terrain = DummyTerrain(has_los=True)

    def bench_trajectory_call() -> None:
        for t in times:
            trajectory(t)

    def bench_spline_scalar() -> None:
        for ts in timestamps:
            spline(ts)

    def bench_spline_scipy_scalar() -> None:
        for ts in timestamps:
            scipy_spline(ts)

    def bench_spline_scipy_vectorized() -> None:
        scipy_spline(timestamps_array)

    def bench_target_validated() -> None:
        for _ in range(BATCH_SIZE):
            Target(
                id=trajectory.target_id,
                is_stationary=False,
                info=trajectory.target_info,
                point=point,
                cross_section_model=trajectory.cross_section_model,
                velocity=velocity,
            )

    def bench_target_model_construct() -> None:
        for _ in range(BATCH_SIZE):
            Target.model_construct(
                id=trajectory.target_id,
                is_stationary=False,
                info=trajectory.target_info,
                point=point,
                cross_section_model=trajectory.cross_section_model,
                velocity=velocity,
            )

    def bench_intersects_terrain() -> None:
        for _ in range(N_TERRAIN_CHECKS):
            trajectory_intersects_terrain(trajectory, terrain)

    return {
        "trajectory_call": (bench_trajectory_call, BATCH_SIZE),
        "spline_scalar": (bench_spline_scalar, BATCH_SIZE),
        "spline_scipy_scalar": (bench_spline_scipy_scalar, BATCH_SIZE),
        "spline_scipy_vectorized": (bench_spline_scipy_vectorized, BATCH_SIZE),
        "target_validated": (bench_target_validated, BATCH_SIZE),
        "target_model_construct": (bench_target_model_construct, BATCH_SIZE),
        "intersects_terrain": (bench_intersects_terrain, N_TERRAIN_CHECKS),
    }


def _run_one(name: str, batch_call: Callable[[], None], batch_size: int) -> None:
    batch_call()  # warm-up: exclude JIT compilation / import-time setup
    best = min(timeit.repeat(batch_call, repeat=TIMEIT_REPEAT, number=1))
    per_call_us = best / batch_size * 1e6
    print(f"{name:35s} {per_call_us:12.3f} us/call  {batch_size / best:14,.0f} calls/s")


def run_all() -> None:
    print(
        f"batch_size={BATCH_SIZE}  "
        f"best-of-{TIMEIT_REPEAT} (timeit.repeat, number=1)\n"
    )
    for name, (fn, batch_size) in build_benchmarks().items():
        _run_one(name, fn, batch_size)


def run_profile(name: str, n_batches: int = 20) -> None:
    benchmarks = build_benchmarks()
    if name not in benchmarks:
        raise SystemExit(f"Unknown benchmark {name!r}. Choices: {list(benchmarks)}")
    fn, _ = benchmarks[name]
    fn()  # warm-up

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(n_batches):
        fn()
    profiler.disable()

    print(f"cProfile over {name} ({n_batches} batches):\n")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        metavar="METHOD",
        help="Run cProfile over the given benchmark instead of timing it. "
        "Choices: trajectory_call, spline_scalar, spline_scipy_scalar, "
        "spline_scipy_vectorized, "
        "target_validated, target_model_construct, intersects_terrain",
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(args.profile)
    else:
        run_all()
