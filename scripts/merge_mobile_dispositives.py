import argparse
import pathlib

from theia.simulation.scenario_import import MobileDispositive

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mobile_deployment1", type=pathlib.Path)
    parser.add_argument("mobile_deployment2", type=pathlib.Path)
    parser.add_argument("output_file", type=pathlib.Path)
    args = parser.parse_args()

    deployment1 = MobileDispositive.from_file(args.mobile_deployment1)
    deployment2 = MobileDispositive.from_file(args.mobile_deployment2)

    result = MobileDispositive.merge(deployment1, deployment2)

    with open(args.output_file, "w") as file:
        file.write(result.model_dump_json(indent=4))
