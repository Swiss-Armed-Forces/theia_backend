import argparse
import math
import pathlib

from tqdm import tqdm

from theia.simulation.scenario_import import ScenarioFactory


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

    simulator = scenario.to_simulator("test.json")

    n_iterations = math.ceil(
        (scenario.stop_time - scenario.start_time) / scenario.time_step
    )

    for _ in tqdm(range(n_iterations + 1)):
        simulator.advance()
