"""
Quick benchmark harness for theia.coordinates.CoordinateTransformations.

Usage
-----
    python scripts/benchmark_coordinate_transformations.py
    python scripts/benchmark_coordinate_transformations.py --profile geodetic_to_cartesian

Design
------
Each method is timed over a fixed batch of BATCH_SIZE precomputed, randomly
sampled inputs (not a single repeated input), so branch prediction / caching
behave roughly like real (varied) call patterns rather than best-casing a
single hot value. Timing uses timeit.repeat and reports the best-of-N total,
which is the standard way to suppress OS/scheduler noise in microbenchmarks.

A warm-up call is made before timing so an implementation that JIT-compiles
on first use (e.g. numba @njit) doesn't have its one-time compilation cost
folded into the measurement.

--profile runs cProfile over the batch instead of timing it, for finding
hotspots to optimize (equally, py-spy/snakeviz -- both already dev
dependencies -- can be attached to a run of this script for a flamegraph).
"""

import argparse
import cProfile
import pstats
from typing import Callable

import numpy as np
import timeit

from theia.coordinates import CoordinateTransformations
from theia.types import Point, Velocity

SEED = 20260821
BATCH_SIZE = 2_000
TIMEIT_REPEAT = 7


def _sample_geodetic_batch(
    n: int, rng: np.random.Generator
) -> list[tuple[float, float, float]]:
    lat = rng.uniform(-90, 90, size=n)
    lon = rng.uniform(-180, 180, size=n)
    alt = rng.uniform(-1_000, 50_000, size=n)
    return list(zip(lat.tolist(), lon.tolist(), alt.tolist()))


def _sample_cartesian_batch(
    n: int, rng: np.random.Generator
) -> list[tuple[float, float, float]]:
    return [
        CoordinateTransformations.geodetic_to_cartesian(lat, lon, alt)
        for lat, lon, alt in _sample_geodetic_batch(n, rng)
    ]


def _sample_points_and_velocities(
    n: int, rng: np.random.Generator
) -> tuple[list[Point], list[Velocity]]:
    points = [
        Point(lat=lat, lon=lon, alt=alt)
        for lat, lon, alt in _sample_geodetic_batch(n, rng)
    ]
    v = rng.uniform(-600, 600, size=(n, 3))
    velocities = [Velocity(vx=x, vy=y, vz=z) for x, y, z in v]
    return points, velocities


def build_benchmarks() -> dict[str, Callable[[], None]]:
    rng = np.random.Generator(np.random.PCG64(seed=SEED))
    geodetic_batch = _sample_geodetic_batch(BATCH_SIZE, rng)
    cartesian_batch = _sample_cartesian_batch(BATCH_SIZE, rng)
    points, velocities = _sample_points_and_velocities(BATCH_SIZE, rng)
    cartesian_velocities = [
        CoordinateTransformations.velocity_geodetic_to_cartesian(p, *v.as_tuple())
        for p, v in zip(points, velocities)
    ]

    def bench_geodetic_to_cartesian() -> None:
        for lat, lon, alt in geodetic_batch:
            CoordinateTransformations.geodetic_to_cartesian(lat, lon, alt)

    def bench_cartesian_to_geodetic() -> None:
        for x, y, z in cartesian_batch:
            CoordinateTransformations.cartesian_to_geodetic(x, y, z)

    def bench_velocity_geodetic_to_cartesian() -> None:
        for p, v in zip(points, velocities):
            CoordinateTransformations.velocity_geodetic_to_cartesian(p, *v.as_tuple())

    def bench_velocity_cartesian_to_geodetic() -> None:
        for p, v in zip(points, cartesian_velocities):
            CoordinateTransformations.velocity_cartesian_to_geodetic(p, v)

    return {
        "geodetic_to_cartesian": bench_geodetic_to_cartesian,
        "cartesian_to_geodetic": bench_cartesian_to_geodetic,
        "velocity_geodetic_to_cartesian": bench_velocity_geodetic_to_cartesian,
        "velocity_cartesian_to_geodetic": bench_velocity_cartesian_to_geodetic,
    }


def _run_one(name: str, batch_call: Callable[[], None]) -> None:
    batch_call()  # warm-up: exclude JIT compilation / import-time setup
    best = min(timeit.repeat(batch_call, repeat=TIMEIT_REPEAT, number=1))
    per_call_us = best / BATCH_SIZE * 1e6
    print(f"{name:35s} {per_call_us:10.3f} us/call  {BATCH_SIZE / best:14,.0f} calls/s")


def run_all() -> None:
    print(
        f"batch_size={BATCH_SIZE}  "
        f"best-of-{TIMEIT_REPEAT} (timeit.repeat, number=1)\n"
    )
    for name, fn in build_benchmarks().items():
        _run_one(name, fn)


def run_profile(name: str, n_batches: int = 20) -> None:
    benchmarks = build_benchmarks()
    if name not in benchmarks:
        raise SystemExit(f"Unknown benchmark {name!r}. Choices: {list(benchmarks)}")
    fn = benchmarks[name]
    fn()  # warm-up

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(n_batches):
        fn()
    profiler.disable()

    print(f"cProfile over {name} ({n_batches} x {BATCH_SIZE} calls):\n")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        metavar="METHOD",
        help="Run cProfile over the given benchmark instead of timing it. "
        "Choices: geodetic_to_cartesian, cartesian_to_geodetic, "
        "velocity_geodetic_to_cartesian, velocity_cartesian_to_geodetic",
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(args.profile)
    else:
        run_all()
