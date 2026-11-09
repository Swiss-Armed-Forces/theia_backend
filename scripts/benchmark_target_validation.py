"""
Quick benchmark harness for theia.types.Target validation, in particular the
legacy-SIDC "before" model validator Target._from_legacy_sidc.

Usage
-----
    python scripts/benchmark_target_validation.py
    python scripts/benchmark_target_validation.py --profile validate_dict

Design
------
See benchmark_coordinate_transformations.py for the general approach (fixed
randomly-sampled batch, timeit.repeat best-of-N, warm-up call).

A mode="before" model validator makes pydantic-core call back into Python for
every Target that is built, on top of the field validation itself. Targets
are built on two hot paths, benchmarked separately:
  - construction from keyword arguments with model instances as values
    (Trajectory.__call__, controllers) -> construct_kwargs
  - validation of plain dicts read from JSON (LogLoader) -> validate_dict
Each is paired with a *_no_validator variant: a copy of Target with the same
fields but without its model validator, so the difference is the overhead
of _from_legacy_sidc. validator_* call the function directly to show its own
cost without pydantic's callback overhead; validate_legacy_dict exercises the
conversion path for logs written before TargetInfo existed.

model_dump (FileLogger.on_snapshot) is included because the same refactoring
added the computed field Target.sidc, which is evaluated on every dump; the
*_no_validator copy has no computed field, so the difference is its cost.
"""

import argparse
import cProfile
import pstats
from typing import Any, Callable

import numpy as np
import pydantic
import timeit

from theia.types import (
    ConstantRcsModel,
    Party,
    Point,
    Target,
    TargetInfos,
    Velocity,
)

SEED = 20261030
BATCH_SIZE = 2_000
TIMEIT_REPEAT = 7

TargetWithoutValidator = pydantic.create_model(
    "TargetWithoutValidator",
    **{name: (field.annotation, field) for name, field in Target.model_fields.items()},
)
"""Same fields as Target, but no model validators and no computed fields."""


def _sample_kwargs(n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    lat = rng.uniform(45.0, 48.0, size=n)
    lon = rng.uniform(5.0, 11.0, size=n)
    alt = rng.uniform(0.0, 12_000.0, size=n)
    v = rng.uniform(-300.0, 300.0, size=(n, 3))
    info = TargetInfos.FIXED_WING.with_party(Party.RED)
    rcs = ConstantRcsModel(rcs=0.5)
    return [
        {
            "id": i,
            "is_stationary": False,
            "info": info,
            "point": Point(lat=float(lat[i]), lon=float(lon[i]), alt=float(alt[i])),
            "cross_section_model": rcs,
            "velocity": Velocity(vx=float(v[i, 0]), vy=float(v[i, 1]), vz=float(v[i, 2])),
        }
        for i in range(n)
    ]


def _to_legacy_dict(target_dict: dict[str, Any]) -> dict[str, Any]:
    """Format of logs written before TargetInfo existed: only a concrete SIDC."""
    legacy = dict(target_dict)
    del legacy["info"]
    del legacy["is_damaged"]
    return legacy


def build_benchmarks() -> dict[str, Callable[[], None]]:
    rng = np.random.Generator(np.random.PCG64(seed=SEED))
    kwargs_batch = _sample_kwargs(BATCH_SIZE, rng)
    targets = [Target(**kw) for kw in kwargs_batch]
    targets_without_validator = [TargetWithoutValidator(**kw) for kw in kwargs_batch]
    dict_batch = [t.model_dump(mode="json") for t in targets]
    legacy_dict_batch = [_to_legacy_dict(d) for d in dict_batch]

    def bench_validator_direct() -> None:
        for d in dict_batch:
            Target._from_legacy_sidc(d)

    def bench_validator_direct_legacy() -> None:
        for d in legacy_dict_batch:
            Target._from_legacy_sidc(d)

    def bench_construct_kwargs() -> None:
        for kw in kwargs_batch:
            Target(**kw)

    def bench_construct_kwargs_no_validator() -> None:
        for kw in kwargs_batch:
            TargetWithoutValidator(**kw)

    def bench_validate_dict() -> None:
        for d in dict_batch:
            Target.model_validate(d)

    def bench_validate_dict_no_validator() -> None:
        for d in dict_batch:
            TargetWithoutValidator.model_validate(d)

    def bench_validate_legacy_dict() -> None:
        for d in legacy_dict_batch:
            Target.model_validate(d)

    def bench_model_dump() -> None:
        for t in targets:
            t.model_dump(mode="json")

    def bench_model_dump_no_computed_sidc() -> None:
        for t in targets_without_validator:
            t.model_dump(mode="json")

    return {
        "validator_direct": bench_validator_direct,
        "validator_direct_legacy": bench_validator_direct_legacy,
        "construct_kwargs": bench_construct_kwargs,
        "construct_kwargs_no_validator": bench_construct_kwargs_no_validator,
        "validate_dict": bench_validate_dict,
        "validate_dict_no_validator": bench_validate_dict_no_validator,
        "validate_legacy_dict": bench_validate_legacy_dict,
        "model_dump": bench_model_dump,
        "model_dump_no_computed_sidc": bench_model_dump_no_computed_sidc,
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
        "Choices: validator_direct, validator_direct_legacy, construct_kwargs, "
        "construct_kwargs_no_validator, validate_dict, validate_dict_no_validator, "
        "validate_legacy_dict, model_dump, model_dump_no_computed_sidc",
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(args.profile)
    else:
        run_all()
