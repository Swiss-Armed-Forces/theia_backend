import argparse
import pathlib

from theia.simulation.scenario_import import (
    Dispositive,
    MobileDispositive,
    ScenarioFactory,
    StaticDispositive,
)
from theia.types import IdProvider


def load_dispositive(
    static_controller_file: str,
    mobile_controller_file: str,
) -> Dispositive:
    static_dispo = StaticDispositive.from_file(static_controller_file)
    mobile_dispo = MobileDispositive.from_file(mobile_controller_file)
    return Dispositive(
        static_dispositive=static_dispo,
        mobile_dispositive=mobile_dispo,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Theia scenario file generator",
        description="Merge dispositive files for BLUE and RED forces into a single scenario file",
    )
    parser.add_argument("scenario_name", type=str)
    parser.add_argument("static_dispo_file_blue", type=pathlib.Path)
    parser.add_argument("mobile_dispo_file_blue", type=pathlib.Path)
    parser.add_argument("static_dispo_file_red", type=pathlib.Path)
    parser.add_argument("mobile_dispo_file_red", type=pathlib.Path)
    parser.add_argument("output_file", type=pathlib.Path)

    args = parser.parse_args()

    id_provider = IdProvider()

    dispo_blue = load_dispositive(
        args.static_dispo_file_blue,
        args.mobile_dispo_file_blue,
    )
    dispo_red = load_dispositive(
        args.static_dispo_file_red,
        args.mobile_dispo_file_red,
    )

    scenario = ScenarioFactory(
        name=args.scenario_name,
        blue_dispositive=dispo_blue,
        red_dispositive=dispo_red,
    )

    with open(args.output_file, "w") as file:
        file.write(scenario.model_dump_json(indent=4))
