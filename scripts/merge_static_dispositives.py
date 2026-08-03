import argparse
import pathlib

from theia.simulation.scenario_import import StaticDispositive

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("static_deployment1", type=pathlib.Path)
    parser.add_argument("static_deployment2", type=pathlib.Path)
    parser.add_argument("output_file", type=pathlib.Path)
    args = parser.parse_args()

    deployment1 = StaticDispositive.from_file(args.static_deployment1)
    deployment2 = StaticDispositive.from_file(args.static_deployment2)

    result = StaticDispositive.merge(deployment1, deployment2)

    with open(args.output_file, "w") as file:
        file.write(result.model_dump_json(indent=4))
