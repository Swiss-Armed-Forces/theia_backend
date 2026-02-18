import copy
from typing import Literal
from theia.coordinates import CoordinateTransformations
from theia.terrain import elevationAt
from theia.types import (
    ConstantRcsModel,
    Point,
    Polarization,
    Radar,
    Receiver,
    Situation,
    Target,
    Transmitter,
)


class TestSituationLoader:
    @staticmethod
    def load_zuerich_single_target(
        speed: float,
        move_direction: Literal["north"]
        | Literal["east"]
        | Literal["south"]
        | Literal["west"] = "south",
    ) -> Situation:
        radar_lat = 47.349491
        radar_lon = 8.492063
        point = Point(
            lat=radar_lat, lon=radar_lon, alt=elevationAt(radar_lat, radar_lon)
        )

        antenna_height = 10.0
        antenna_diameter = 2.0

        radar = Radar(
            transmitter=Transmitter(
                id=0,
                point=point,
                power=1_000,
                erp=1000.0,
                antenna_height=antenna_height,
                antenna_diameter=antenna_diameter,
                frequency=1000.0,
                pulse_width=1.0,
                polarization=Polarization.VERTICAL,
                bandwidth=100,
            ),
            receiver=Receiver(
                id=0,
                point=point,
                antenna_height=antenna_height,
                diameter=antenna_diameter,
                cpi_pulses=1,
                pfa=1e-6,
                min_elevation=-20.0,
                max_elevation=60.0,
                rotation_time=10,
                bandwidth=100.0,
            ),
        )

        p_target = copy.deepcopy(point)
        p_target.lat += 0.1

        if move_direction == "north":
            v = CoordinateTransformations.velocity_geodetic_to_cartesian(
                p_target, -speed, 0.0, 0.0
            )
        elif move_direction == "east":
            v = CoordinateTransformations.velocity_geodetic_to_cartesian(
                p_target, 0.0, speed, 0.0
            )
        elif move_direction == "south":
            v = CoordinateTransformations.velocity_geodetic_to_cartesian(
                p_target, speed, 0.0, 0.0
            )
        elif move_direction == "west":
            v = CoordinateTransformations.velocity_geodetic_to_cartesian(
                p_target, 0.0, -speed, 0.0
            )
        else:
            raise RuntimeError(f"Unknown direction: {move_direction}")

        target = Target(
            id=0,
            point=p_target,
            cross_section_model=ConstantRcsModel(rcs=2.0),
            velocity=v,
        )

        return Situation(
            radars=[radar],
            targets=[target],
            transmitter_labels={0: "Tx"},
            target_labels={0: "Target (southbound)"},
        )
