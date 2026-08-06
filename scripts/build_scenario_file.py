import argparse
import datetime
import pathlib

import numpy as np

from theia.coordinates import POSITIONS_OF_INTEREST
from theia.simulation.scenario_import import (
    DamageModelFactory,
    OrderOfBattle,
    PseudoTrackerParams,
    ScenarioFactory,
    TerrainFactory,
    TrackerFactory,
)
from theia.types import IdProvider, Point


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Theia scenario file generator",
        description="Merge dispositive files for BLUE and RED forces into a single scenario file",
    )
    parser.add_argument("scenario_name", type=str)
    parser.add_argument("orbat_file_blue", type=pathlib.Path)
    parser.add_argument("orbat_file_red", type=pathlib.Path)
    parser.add_argument("output_file", type=pathlib.Path)
    parser.add_argument("--terrain_model", type=str, default="SRTM")

    args = parser.parse_args()

    terrain_factory = TerrainFactory(terrain_name=args.terrain_model)
    terrain = terrain_factory.to_terrain()

    orbat_blue = OrderOfBattle.from_file(args.orbat_file_blue)
    orbat_red = OrderOfBattle.from_file(args.orbat_file_red)

    id_provider = IdProvider()
    orbat_blue.update_id_provider(id_provider)
    orbat_red.reindex(id_provider)

    t_min = None
    t_max = None
    if orbat_blue.t_min is not None:
        t_min = orbat_blue.t_min
        t_max = orbat_blue.t_max
    if orbat_red.t_min is not None:
        t_min = orbat_red.t_min if t_min is None else min(t_min, orbat_red.t_min)
        t_max = orbat_red.t_max if t_max is None else max(t_max, orbat_red.t_max)

    if t_min is None or t_max is None:
        raise ValueError("No mobile parts in the simulation!")

    t_max = datetime.datetime.fromtimestamp(
        np.ceil(t_max.timestamp()),
        datetime.UTC,
    )

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
        blue_orbat=orbat_blue,
        red_orbat=orbat_red,
        blue_tracker=TrackerFactory(
            tracker_name="pseudotracker",
            parameters=tracker_params,
        ),
        red_tracker=TrackerFactory(
            tracker_name="pseudotracker",
            parameters=tracker_params,
        ),
        terrain_model=terrain_factory,
        damage_model=DamageModelFactory(model_name="kill_always"),
        seed=39270507,
    )

    with open(args.output_file, "w") as file:
        file.write(scenario.model_dump_json(indent=4))
