import datetime
import pathlib

import numpy as np

from theia.config import SIDC
from theia.distance import burstvincentydistance
from theia.simulation.scenario_import import (
    DirectFireEffectorFactory,
    FixedPathOneWayDroneFactory,
    MobileDispositive,
    TerrainFactory,
)
from theia.terrain import SrtmTerrainModel
from theia.types import ConstantRcsModel, Entity, IdProvider, Point, Trajectory


def build_single_drone(
    id_provider: IdProvider,
    p_target: Point,
    bearing: float,  # deg
    t_start: datetime.datetime,
    t_stop_sim: datetime.datetime,
    alt: float = 1000.0,
    d_start: float = 150_000.0,
    d_descent: float = 10_000.0,
    speed: float = 300,  # m / s
    effect_radius: float = 100.0,
    name: str = "",
) -> FixedPathOneWayDroneFactory:
    points: list[Point] = []
    times: list[datetime.datetime] = []
    # Start at d_start from the airport and move toward the airport.
    # Travel at altitude alt until you reach d_descent, then descend linearly
    # to the airport.
    for i, d in enumerate(np.arange(0.0, d_start, step=speed)[::-1]):
        t = t_start + datetime.timedelta(seconds=i)
        if t >= t_stop_sim:
            break
        p = burstvincentydistance(
            (p_target.lat, p_target.lon),
            d,
            bearing,
            alt
            if d > d_descent
            else p_target.alt + d * (alt - p_target.alt) / d_descent,
        )
        points.append(p)
        times.append(t)
    # The drones that arrive at the target point ad disappear.

    trajectory = Trajectory(
        target_id=id_provider.increment(Entity.TARGET, name),
        target_sidc=SIDC.RED_FIXED_WING,
        times=times,
        lats=[p.lat for p in points],
        lons=[p.lon for p in points],
        alts=[p.alt for p in points],
        vxs=[0.0 for _ in range(len(points))],
        vys=[0.0 for _ in range(len(points))],
        vzs=[0.0 for _ in range(len(points))],
        cross_section_model=ConstantRcsModel(rcs=1.0),
    )
    terrain = TerrainFactory(terrain_name="SRTM")
    return FixedPathOneWayDroneFactory(
        effector=DirectFireEffectorFactory(
            id=id_provider.increment(Entity.EFFECTOR, name),
            name=name,
            point=trajectory(trajectory.times[0]).point,
            combat_range=effect_radius,
            n_attacks_left=1,
            terrain=terrain,
        ),
        trajectory=trajectory,
        assigned_goal=p_target,
        terrain=terrain,
    )


def build_dispositive(
    rng: np.random.Generator,
    id_provider: IdProvider,
    p_target: Point,
    n_attackers: int = 100,
    t_max_s: int = 600,
    start_distance: float = 150_000.0,
    descent_distance: float = 10_000.0,
    speed: float = 300.0,
    effect_radius: float = 100.0,
) -> MobileDispositive:
    angles = rng.uniform(0, 90, size=(n_attackers,))
    # Hardcode t = 0: Start the simulation at zero.
    offset_seconds = [0] + rng.uniform(0, t_max_s, size=(n_attackers - 1,)).tolist()

    t0 = datetime.datetime.fromtimestamp(0, tz=datetime.UTC)
    tmax = datetime.datetime.fromtimestamp(t_max_s, tz=datetime.UTC)

    id_provider = IdProvider()
    factories: list[FixedPathOneWayDroneFactory] = []
    for i, (angle, t_start_seconds) in enumerate(
        zip(angles, offset_seconds, strict=True)
    ):
        factories.append(
            build_single_drone(
                id_provider,
                p_target,
                angle,
                t0 + datetime.timedelta(seconds=t_start_seconds),
                tmax,
                1000.0,
                start_distance,
                descent_distance,
                speed,
                effect_radius,
                name=f"drone {i}",
            )
        )
    dispo = MobileDispositive(oneway_drones=factories)
    return dispo


if __name__ == "__main__":
    N_drones = 100

    rng = np.random.default_rng(seed=3427087)
    id_provider = IdProvider()

    terrain_model = SrtmTerrainModel()

    BLUE_INFRA_LAT = 47.45270
    BLUE_INFRA_LON = 8.56068
    p_airport = Point(
        lat=BLUE_INFRA_LAT,
        lon=BLUE_INFRA_LON,
        alt=terrain_model.elevationAt(BLUE_INFRA_LAT, BLUE_INFRA_LON) + 10,
    )

    dispo = build_dispositive(
        rng,
        id_provider,
        p_airport,
        n_attackers=N_drones,
        t_max_s=600,
    )
    script_dir = pathlib.Path(__file__).parent
    with open(
        f"{script_dir}/../scenarios/oneway_attack_ZRH_{N_drones}drones_10min.mobile_dispositive.json",
        "w",
    ) as file:
        file.write(dispo.model_dump_json(indent=4))
