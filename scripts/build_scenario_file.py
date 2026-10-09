import argparse
import pathlib

from theia.simulation.scenario_import import (
    OrderOfBattle,
    TerrainFactory,
    build_scenario,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Theia scenario file generator",
        description="Merge dispositive files for BLUE and RED forces into a single scenario file",
    )
    parser.add_argument("scenario_name", type=str)
    parser.add_argument("orbat_file_blue", type=pathlib.Path)
    parser.add_argument("orbat_file_red", type=pathlib.Path)
    parser.add_argument("output_file", type=pathlib.Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("--terrain_model", type=str, default="SRTM")

    args = parser.parse_args()

    terrain_factory = TerrainFactory(terrain_name=args.terrain_model)
    terrain = terrain_factory.to_terrain()

    orbat_blue = OrderOfBattle.from_file(args.orbat_file_blue)
    orbat_red = OrderOfBattle.from_file(args.orbat_file_red)

    scenario = build_scenario(
        args.scenario_name,
        terrain_factory,
        args.seed,
        orbat_blue,
        orbat_red,
    )

    with open(args.output_file, "w") as file:
        file.write(scenario.model_dump_json(indent=4))
