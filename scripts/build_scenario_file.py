import argparse
import math
import pathlib

from theia.coordinates import POSITIONS_OF_INTEREST
from theia.simulation.scenario_import import (
    DamageModelFactory,
    Dispositive,
    MobileDispositive,
    PseudoTrackerParams,
    ScenarioFactory,
    StaticDispositive,
    TerrainFactory,
    TrackerFactory,
)
from theia.types import IdProvider, Point


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

    t_min = None
    t_max = None
    if dispo_blue.mobile_dispositive.t_min is not None:
        t_min = dispo_blue.mobile_dispositive.t_min
        t_max = dispo_blue.mobile_dispositive.t_max
    if dispo_red.mobile_dispositive.t_min is not None:
        t_min = (
            dispo_red.mobile_dispositive.t_min
            if t_min is None
            else min(t_min, dispo_red.mobile_dispositive.t_min)
        )
        t_max = (
            dispo_red.mobile_dispositive.t_max
            if t_max is None
            else min(t_max, dispo_red.mobile_dispositive.t_max)
        )

    if t_min is None or t_max is None:
        raise ValueError("No mobile parts in the simulation!")

    tracker_params = PseudoTrackerParams(
        removal_patience=10,
        start_timestamp=t_min.timestamp(),
        prior_position=Point(
            lat=POSITIONS_OF_INTEREST["CH_CENTER"]["lat"],
            lon=POSITIONS_OF_INTEREST["CH_CENTER"]["lon"],
            alt=1000,
        ),
    )

    scenario = ScenarioFactory(
        name=args.scenario_name,
        start_time=t_min,
        stop_time=t_max,
        time_step=1,
        blue_dispositive=dispo_blue,
        red_dispositive=dispo_red,
        blue_tracker=TrackerFactory(
            tracker_name="pseudotracker",
            parameters=tracker_params,
        ),
        red_tracker=TrackerFactory(
            tracker_name="pseudotracker",
            parameters=tracker_params,
        ),
        terrain_model=TerrainFactory(terrain_name="SRTM"),
        damage_model=DamageModelFactory(model_name="kill_always"),
        seed=39270507,
    )

    with open(args.output_file, "w") as file:
        file.write(scenario.model_dump_json(indent=4))
