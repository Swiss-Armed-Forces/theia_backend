"""
Maximum speedup and profile of the game demo without the server.

Usage
-----
    python scripts/profile_game_demo.py
    python scripts/profile_game_demo.py --warmup 300 --steps 1000 --prof out.prof

Builds scenarios/player_actions_test as in scripts/simulate_scenario.py
(non-interactive, so no sleeping between steps), advances it ``--warmup``
steps, times ``--steps`` steps and then profiles another ``--steps`` steps
with cProfile. The time per step must not grow with the game time; run with
a larger ``--warmup`` to check.

Prints the speedup (game time per wall-clock time) of the unprofiled steps and
the profile, sorted by cumulative time (Theia functions only) and by internal
time. cProfile roughly doubles the times: compare shares, not absolute times.
"""

import argparse
import cProfile
import pathlib
import pstats
import tempfile
import time

from theia.simulation.scenario_import import ScenarioFactory

SCENARIO = (
    pathlib.Path(__file__).parent.parent
    / "scenarios"
    / "player_actions_test"
    / "player_actions_test.json"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--prof", type=pathlib.Path, help="Save the profile here")
    args = parser.parse_args()

    scenario = ScenarioFactory.model_validate_json(SCENARIO.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        simulator, _ = scenario.to_simulator(
            str(pathlib.Path(tmp) / "log.json"), False, False
        )

        for _ in range(args.warmup):
            simulator.advance()

        start = time.perf_counter()
        for _ in range(args.steps):
            simulator.advance()
        seconds_per_step = (time.perf_counter() - start) / args.steps
        print(
            f"{seconds_per_step * 1000:.2f} ms/step, "
            f"speedup {scenario.time_step / seconds_per_step:.0f}x "
            f"(after {args.warmup} steps of warm-up)"
        )

        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.steps):
            simulator.advance()
        profiler.disable()

    if args.prof is not None:
        profiler.dump_stats(args.prof)
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats("theia", 30)
    stats.sort_stats("tottime").print_stats(15)
