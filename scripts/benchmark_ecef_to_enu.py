"""
Quick benchmark harness for theia.coordinates.EcefToEnuTransformer.

Usage
-----
    python scripts/benchmark_ecef_to_enu.py
    python scripts/benchmark_ecef_to_enu.py --profile construct_and_ecef_to_enu

Design
------
See benchmark_coordinate_transformations.py for the general approach (fixed
randomly-sampled batch, timeit.repeat best-of-N, warm-up call).

EcefToEnuTransformer is used in two distinct real patterns in this codebase:
  - construct once, transform many points (grids.py, terrain_fast_los.py)
  - construct fresh and transform exactly one point, every call
    (measurement.py's MonostaticMeasurementTransformations -- called once per
    detection, so construction cost is *not* amortized there)
Both are benchmarked separately below, since an optimization that only helps
one of them (e.g. speeding up repeated ecef_to_enu on a fixed transformer
without touching __init__) would be invisible on the other.
"""

import argparse
import cProfile
import pstats
from typing import Callable

import numpy as np
import timeit

from theia.coordinates import EcefToEnuTransformer
from theia.types import Point

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


def _sample_points(n: int, rng: np.random.Generator) -> list[Point]:
    return [
        Point(lat=lat, lon=lon, alt=alt)
        for lat, lon, alt in _sample_geodetic_batch(n, rng)
    ]


def _sample_enu_batch(
    n: int, rng: np.random.Generator
) -> list[tuple[float, float, float]]:
    v = rng.uniform(-1_000_000, 1_000_000, size=(n, 3))
    return [tuple(row) for row in v.tolist()]


def build_benchmarks() -> dict[str, Callable[[], None]]:
    rng = np.random.Generator(np.random.PCG64(seed=SEED))
    reference_points = _sample_points(BATCH_SIZE, rng)
    target_points = _sample_points(BATCH_SIZE, rng)

    fixed_reference = reference_points[0]
    fixed_transformer = EcefToEnuTransformer(fixed_reference)

    ecef_batch = [
        fixed_transformer.enu_to_ecef((float(i), float(-i), float(i % 7)))
        for i in range(BATCH_SIZE)
    ]
    enu_batch = _sample_enu_batch(BATCH_SIZE, rng)

    def bench_construct() -> None:
        for ref in reference_points:
            EcefToEnuTransformer(ref)

    def bench_ecef_to_enu() -> None:
        for p in ecef_batch:
            fixed_transformer.ecef_to_enu(p)

    def bench_enu_to_ecef() -> None:
        for p in enu_batch:
            fixed_transformer.enu_to_ecef(p)

    def bench_ecef_to_enu_multiple() -> None:
        fixed_transformer.ecef_to_enu_multiple(ecef_batch)

    def bench_construct_and_ecef_to_enu() -> None:
        # Matches MonostaticMeasurementTransformations.cartesian_to_elevation_azimuth_range:
        # a fresh transformer per call, one ecef_to_enu call.
        for ref, p in zip(reference_points, ecef_batch):
            EcefToEnuTransformer(ref).ecef_to_enu(p)

    def bench_construct_and_enu_to_ecef() -> None:
        # Matches MonostaticMeasurementTransformations.elevation_azimuth_range_to_ecef:
        # a fresh transformer per call, one enu_to_ecef call.
        for ref, p in zip(reference_points, enu_batch):
            EcefToEnuTransformer(ref).enu_to_ecef(p)

    return {
        "construct": bench_construct,
        "ecef_to_enu": bench_ecef_to_enu,
        "enu_to_ecef": bench_enu_to_ecef,
        "ecef_to_enu_multiple": bench_ecef_to_enu_multiple,
        "construct_and_ecef_to_enu": bench_construct_and_ecef_to_enu,
        "construct_and_enu_to_ecef": bench_construct_and_enu_to_ecef,
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
        "Choices: construct, ecef_to_enu, enu_to_ecef, ecef_to_enu_multiple, "
        "construct_and_ecef_to_enu, construct_and_enu_to_ecef",
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(args.profile)
    else:
        run_all()
