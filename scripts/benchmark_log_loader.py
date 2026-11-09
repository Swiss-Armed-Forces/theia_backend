"""
Quick benchmark harness for theia.simulation.theia_logging.LogLoader.

Usage
-----
    python scripts/benchmark_log_loader.py LOG_FILE
    python scripts/benchmark_log_loader.py LOG_FILE --profile load_snapshots

LOG_FILE is a log written by FileLogger, e.g. run_000.log.json from the
mission_simulation notebook or scripts/simulate_scenario.py.

Design
------
Unlike the other benchmark_*.py scripts, there is no random batch: the input
is one real log file, and each benchmark processes it once per call. Timing
is best-of-N (timeit.repeat, number=1) after a warm-up call.

LogLoader parses the JSON eagerly and validates detections and events in
__init__; snapshots and ground truth are validated lazily on first access.
The benchmarks follow that structure, each stage isolated from the ones
before it (stages after json_load reuse the already-parsed data):
  - json_load:               json.load of the file
  - loader_init:             LogLoader(path), i.e. json_load + detections/events
  - load_snapshots:          Snapshot.model_validate for every snapshot
  - validate_targets_only:   the Target part of load_snapshots
  - validate_sensors_only:   the MonostaticSensor/PclSensor part of load_snapshots
  - red_target_ground_truth: GroundTruthPath building from loaded snapshots
  - analyse_run:             end-to-end, as in the notebook's analyse_run()
"""

import argparse
import cProfile
import json
import pathlib
import pstats
from typing import Callable

import pydantic
import timeit

from theia.simulation.theia_logging import LogLoader
from theia.types import MonostaticSensor, PclSensor, Target

TIMEIT_REPEAT = 5

_TARGET_KEYS = ("blue_targets", "red_targets")
_SENSOR_KEYS = (
    "blue_monostatic_radars",
    "blue_pcl_sensors",
    "red_monostatic_radars",
    "red_pcl_sensors",
)


def build_benchmarks(log_file: pathlib.Path) -> dict[str, Callable[[], None]]:
    loader = LogLoader(str(log_file))
    snapshot_dicts = loader._data["snapshots"]
    target_dicts = [t for s in snapshot_dicts for k in _TARGET_KEYS for t in s[k]]
    sensor_dicts = [t for s in snapshot_dicts for k in _SENSOR_KEYS for t in s[k]]
    targets_adapter = pydantic.TypeAdapter(list[Target])
    sensors_adapter = pydantic.TypeAdapter(list[MonostaticSensor | PclSensor])
    loader._snapshots_loaded = loader._load_snapshots()

    print(
        f"{log_file}: {log_file.stat().st_size / 1e6:.1f} MB, "
        f"{len(snapshot_dicts)} snapshots, {len(target_dicts)} targets, "
        f"{len(sensor_dicts)} sensors, {len(loader._data['detections'])} detection "
        f"entries, {len(loader._data['events'])} events\n"
    )

    def bench_json_load() -> None:
        with open(log_file, "r") as file:
            json.load(file)

    def bench_loader_init() -> None:
        LogLoader(str(log_file))

    def bench_load_snapshots() -> None:
        loader._load_snapshots()

    def bench_validate_targets_only() -> None:
        targets_adapter.validate_python(target_dicts)

    def bench_validate_sensors_only() -> None:
        sensors_adapter.validate_python(sensor_dicts)

    def bench_red_target_ground_truth() -> None:
        loader._load_target_ground_truth(is_blue=False)

    def bench_analyse_run() -> None:
        # Mirrors the accesses of analyse_run() in mission_simulation.ipynb.
        run_loader = LogLoader(str(log_file))
        run_loader.red_target_ids
        run_loader.kill_events
        run_loader.shots

    return {
        "json_load": bench_json_load,
        "loader_init": bench_loader_init,
        "load_snapshots": bench_load_snapshots,
        "validate_targets_only": bench_validate_targets_only,
        "validate_sensors_only": bench_validate_sensors_only,
        "red_target_ground_truth": bench_red_target_ground_truth,
        "analyse_run": bench_analyse_run,
    }


def _run_one(name: str, call: Callable[[], None]) -> None:
    call()  # warm-up: exclude import-time setup
    best = min(timeit.repeat(call, repeat=TIMEIT_REPEAT, number=1))
    print(f"{name:35s} {best * 1e3:10.1f} ms/call")


def run_all(log_file: pathlib.Path) -> None:
    benchmarks = build_benchmarks(log_file)
    print(f"best-of-{TIMEIT_REPEAT} (timeit.repeat, number=1)\n")
    for name, fn in benchmarks.items():
        _run_one(name, fn)


def run_profile(log_file: pathlib.Path, name: str, n_calls: int = 5) -> None:
    benchmarks = build_benchmarks(log_file)
    if name not in benchmarks:
        raise SystemExit(f"Unknown benchmark {name!r}. Choices: {list(benchmarks)}")
    fn = benchmarks[name]
    fn()  # warm-up

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(n_calls):
        fn()
    profiler.disable()

    print(f"cProfile over {name} ({n_calls} calls):\n")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log_file", type=pathlib.Path)
    parser.add_argument(
        "--profile",
        metavar="METHOD",
        help="Run cProfile over the given benchmark instead of timing it. "
        "Choices: json_load, loader_init, load_snapshots, validate_targets_only, "
        "validate_sensors_only, red_target_ground_truth, analyse_run",
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(args.log_file, args.profile)
    else:
        run_all(args.log_file)
