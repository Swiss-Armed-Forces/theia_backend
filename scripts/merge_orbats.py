import argparse
import pathlib

from theia.simulation.scenario_import import OrderOfBattle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("orbat_file1", type=pathlib.Path)
    parser.add_argument("orbat_file2", type=pathlib.Path)
    parser.add_argument("output_file", type=pathlib.Path)
    args = parser.parse_args()

    orbat1 = OrderOfBattle.from_file(args.orbat_file1)
    orbat2 = OrderOfBattle.from_file(args.orbat_file2)

    result = OrderOfBattle.merge(orbat1, orbat2)

    with open(args.output_file, "w") as file:
        file.write(result.model_dump_json(indent=4))
