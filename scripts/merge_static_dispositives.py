import argparse
import pathlib

from theia.simulation.scenario_import import StaticDispositive, TerrainFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("static_deployment1", type=pathlib.Path)
    parser.add_argument("static_deployment2", type=pathlib.Path)
    parser.add_argument("output_file", type=pathlib.Path)
    parser.add_argument("--terrain_model", type=str, default="SRTM")
    args = parser.parse_args()

    terrain_factory = TerrainFactory(terrain_name=args.terrain_model)
    terrain = terrain_factory.to_terrain()

    deployment1 = StaticDispositive.from_file(args.static_deployment1, terrain)
    deployment2 = StaticDispositive.from_file(args.static_deployment2, terrain)

    result = StaticDispositive.merge(deployment1, deployment2)

    with open(args.output_file, "w") as file:
        file.write(result.model_dump_json(indent=4))
