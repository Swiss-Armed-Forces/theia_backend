import argparse
import math
import pathlib

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
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    with open(args.scenario_file, "r") as file:
        scenario = ScenarioFactory.model_validate_json(file.read())

    simulator, buffer = scenario.to_simulator("test.json", args.interactive)

    if args.interactive:
        app = create_app(buffer, SimulationDirector(simulator))

        print("Start simulation...")

        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        n_iterations = math.ceil(
            (scenario.stop_time - scenario.start_time).seconds / scenario.time_step
        )

        for _ in tqdm(range(n_iterations + 1)):
            simulator.advance()
