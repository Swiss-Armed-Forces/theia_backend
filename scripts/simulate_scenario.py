import argparse
import cProfile
import math
import pathlib
import sys

from tqdm import tqdm
import uvicorn

from theia.simulation.scenario_import import ScenarioFactory
from theia.simulation.server import create_app
from theia.simulation.simulation_director import SimulationDirector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Theia scenario simulator",
        description="Simulate a scenario using Theia",
    )
    parser.add_argument("scenario_file", type=pathlib.Path)
    parser.add_argument("output_file", type=pathlib.Path)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debugging", action="store_true")
    args = parser.parse_args()

    if args.interactive and args.profile:
        sys.exit("Interactive and profiling mode cannot be activated at the same time.")

    with open(args.scenario_file, "r") as file:
        scenario = ScenarioFactory.model_validate_json(file.read())

    simulator, buffer = scenario.to_simulator(
        args.output_file,
        args.interactive,
        args.debugging,
    )

    if args.interactive:
        app = create_app(buffer, SimulationDirector(simulator))

        print("Start simulation...")

        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        n_iterations = math.ceil(
            (scenario.stop_time - scenario.start_time).seconds / scenario.time_step
        ) + 1

        if args.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        for _ in tqdm(range(n_iterations)):
            simulator.advance()

        if args.profile:
            profiler.disable()
            profiler.dump_stats(f"profile_{scenario.name}.prof")
