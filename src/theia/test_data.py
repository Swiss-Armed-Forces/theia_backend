import copy
from typing import Literal
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.terrain import elevationAt
from theia.types import (
    ConstantRcsModel,
    MonostaticRadarMeasurementModel,
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
        radar_lat = POSITIONS_OF_INTEREST["Uetliberg"]["lat"]
        radar_lon = POSITIONS_OF_INTEREST["Uetliberg"]["lon"]
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
            error_model=MonostaticRadarMeasurementModel(),
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


def get_uetliberg_radar(
    frequency: float = 3_000,
    power: float = 500_000,
    diameter: float = 4.0,
    bandwidth: float = 5.0,
):
    """
    Parameters
    ----------
    frequency: float, default 3000.0
        Frequency [Hz]
    power: float, default 500000.0
        Power [W]
    diameter: float, default 4.0
        Diameter [m]
    bandwidth: float, default 5.0
        Bandwidth [MHz]
    """

    radar_lat = POSITIONS_OF_INTEREST["Uetliberg"]["lat"]
    radar_lon = POSITIONS_OF_INTEREST["Uetliberg"]["lon"]
    radar_alt = POSITIONS_OF_INTEREST["Uetliberg"]["alt"]

    return Radar(
        transmitter=Transmitter(
            id=0,
            point=Point(lat=radar_lat, lon=radar_lon, alt=radar_alt),
            power=power,
            erp=1000.0,
            antenna_height=10.0,
            antenna_diameter=diameter,
            frequency=frequency,
            pulse_width=1.0,
            polarization=Polarization.VERTICAL,
            bandwidth=bandwidth,
            max_coherent_integration_time=0.5,
            antenna_efficiency_value=0.6,
            vertical_attenuation=None,
            horizontal_attenuation=None,
        ),
        receiver=Receiver(
            id=0,
            point=Point(lat=radar_lat, lon=radar_lon, alt=radar_alt),
            antenna_height=10.0,
            diameter=diameter,
            cpi_pulses=1.0,
            pfa=1e-06,
            min_elevation=-20.0,
            max_elevation=60.0,
            rotation_time=10.0,
            bandwidth=bandwidth,
            gain=0,
            losses=0,
            noise_temperature=300.0,
            noise_figure=1.9,
            antenna_efficiency_value=0.6,
            vertical_attenuation=None,
            horizontal_attenuation=None,
        ),
        error_model=MonostaticRadarMeasurementModel(
            min_range_uncertainty=0.0,
            max_range_uncertainty=0.0,
            min_angular_uncertainty=0.0,
            max_angular_uncertainty=0.0,
        ),
    )
