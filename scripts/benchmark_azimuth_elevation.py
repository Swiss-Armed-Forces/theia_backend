"""
Quick benchmark harness for calculate_azimuth_angle and calculate_elevation_angle.

Usage
-----
    python scripts/benchmark_azimuth_elevation.py
    python scripts/benchmark_azimuth_elevation.py --profile calculate_elevation_angle

Design
------
See benchmark_coordinate_transformations.py for the general approach (fixed
randomly-sampled batch, timeit.repeat best-of-N, warm-up call).

Both functions are called independently, once per call site, in the real
codebase (snr.py's antenna attenuation, detection/pet.py's PET detections,
maneuvers.py) -- never as a paired "give me both angles" call -- so they're
benchmarked separately here, each on its own random (observer, target) batch.
"""

import argparse
import cProfile
import pstats
from typing import Callable

import numpy as np
import timeit

from theia.coordinates import calculate_azimuth_angle, calculate_elevation_angle
from theia.types import Point

SEED = 20260821
BATCH_SIZE = 2_000
TIMEIT_REPEAT = 7


def _sample_points(n: int, rng: np.random.Generator) -> list[Point]:
    lat = rng.uniform(-90, 90, size=n)
    lon = rng.uniform(-180, 180, size=n)
    alt = rng.uniform(-1_000, 50_000, size=n)
    return [
        Point(lat=float(la), lon=float(lo), alt=float(al))
        for la, lo, al in zip(lat, lon, alt)
    ]


def build_benchmarks() -> dict[str, Callable[[], None]]:
    rng = np.random.Generator(np.random.PCG64(seed=SEED))
    observers = _sample_points(BATCH_SIZE, rng)
    targets = _sample_points(BATCH_SIZE, rng)

    def bench_calculate_azimuth_angle() -> None:
        for obs, tgt in zip(observers, targets):
            calculate_azimuth_angle(obs, tgt)

    def bench_calculate_elevation_angle() -> None:
        for obs, tgt in zip(observers, targets):
            calculate_elevation_angle(obs, tgt)

    return {
        "calculate_azimuth_angle": bench_calculate_azimuth_angle,
        "calculate_elevation_angle": bench_calculate_elevation_angle,
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
        "Choices: calculate_azimuth_angle, calculate_elevation_angle",
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(args.profile)
    else:
        run_all()
