from scipy.optimize import bisect
from theia.distance import burstvincentydistance
from theia.terrain import AbstractTerrainModel
from theia.types import Point


def line_of_sight_along_ray(
    emitter_pos: Point,
    target_bearing: float,
    target_alt: float,
    d_max: float,
    dist_res: float,
    terrain_model: AbstractTerrainModel,
) -> Point:
    def loss(distance: float) -> float:
        p = burstvincentydistance(
            (emitter_pos.lat, emitter_pos.lon),
            distance,
            target_bearing,
            target_alt,
        )
        if terrain_model.has_line_of_sight(emitter_pos, p):
            return 1.0
        else:
            return -1.0

    # We need to catch this special case (line of sight at maximum range)
    # because scipy.optimize.bisect expects the extreme values to have different
    # values.
    if loss(d_max) > 0:
        return burstvincentydistance(
            (emitter_pos.lat, emitter_pos.lon),
            d_max,
            target_bearing,
            target_alt,
        )

    d = bisect(loss, 0, d_max, xtol=dist_res)
    return burstvincentydistance(
        (emitter_pos.lat, emitter_pos.lon),
        d,
        target_bearing,
        target_alt,
    )
