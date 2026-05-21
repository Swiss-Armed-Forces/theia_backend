from bayes_opt import BayesianOptimization

from theia.detection.pcl import PclDetector
from theia.terrain import AbstractTerrainModel, SrtmTerrainModel
from theia.test_data import build_pcl_receiver
from theia.types import Point, Receiver, Target, Transmitter


def optimize_pcl_receiver(
    targets: list[Target],
    transmitters: list[Transmitter],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    terrain: AbstractTerrainModel = SrtmTerrainModel(),
) -> tuple[Receiver, BayesianOptimization]:

    def f(lat: float, lon: float, detector: PclDetector = PclDetector()) -> float:
        rx = build_pcl_receiver(
            0,
            Point(lat=lat, lon=lon, alt=terrain.elevationAt(lat, lon)),
        )
        value_sum = 0.0
        for target in targets:
            snrs = []
            for tx in transmitters:
                try:
                    snrs.append(detector.calculate_raw_measurement(rx, tx, target)[0])
                except:
                    pass
            snrs = sorted(snrs)
            value_sum += sum(snrs[-3:])
        return value_sum

    pbounds = {
        "lat": (lat_min, lat_max),
        "lon": (lon_min, lon_max),
    }

    n_init = 30

    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
    )
    optimizer.maximize(
        init_points=n_init,
        n_iter=60,
    )

    p_max = (optimizer.max["params"]["lat"], optimizer.max["params"]["lon"])
    p = Point(
        lat=p_max[0],
        lon=p_max[1],
        alt=terrain.elevationAt(p_max[0], p_max[1]),
    )

    rx = build_pcl_receiver(0, p)

    return rx, optimizer
