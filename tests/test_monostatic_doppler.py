import unittest

from theia.coordinates import CoordinateTransformations
from theia.doppler import calculate_doppler_shift
from theia.test_data import TestSituationLoader
from theia.types import (
    ConstantRcsModel,
    MonostaticRadarMeasurementModel,
    Point,
    Polarization,
    Radar,
    Receiver,
    Target,
    Transmitter,
    calculate_antenna_gain,
)
from theia.util import frequency_to_wavelength


class MonostaticDopplerTest(unittest.TestCase):
    def test_zueri_northbound(self):
        doppler_correct = -1334.2564
        radars, targets = TestSituationLoader.load_zuerich_single_target(
            speed=200.0,
            move_direction="north",
        )
        doppler = calculate_doppler_shift(
            radars[0].receiver,
            targets[0],
            radars[0].transmitter,
        )
        self.assertAlmostEqual(doppler_correct, doppler, delta=1e-2)

    def test_zueri_eastbound(self):
        doppler_correct = 0
        radars, targets = TestSituationLoader.load_zuerich_single_target(
            speed=200.0,
            move_direction="east",
        )
        doppler = calculate_doppler_shift(
            radars[0].receiver,
            targets[0],
            radars[0].transmitter,
        )
        self.assertAlmostEqual(doppler_correct, doppler, delta=1e-2)

    def test_zueri_southbound(self):
        doppler_correct = 1334.2564
        radars, targets = TestSituationLoader.load_zuerich_single_target(
            speed=200.0,
            move_direction="south",
        )
        doppler = calculate_doppler_shift(
            radars[0].receiver,
            targets[0],
            radars[0].transmitter,
        )
        self.assertAlmostEqual(doppler_correct, doppler, delta=1e-2)

    def test_zueri_westbound(self):
        doppler_correct = 0
        radars, targets = TestSituationLoader.load_zuerich_single_target(
            speed=200.0,
            move_direction="west",
        )
        doppler = calculate_doppler_shift(
            radars[0].receiver,
            targets[0],
            radars[0].transmitter,
        )
        self.assertAlmostEqual(doppler_correct, doppler, delta=1e-2)

    def test_monostatic_doppler(self):
        TOLERANCE = 0.7

        correct = -1586.304654593825

        diameter = 2.0
        frequency = 1_000.0
        antenna_efficiency = 0.6

        p = Point(
            lat=47.36700085728634,
            lon=8.537724304199216,
            alt=407.83600886023686,
        )
        radar = Radar(
            transmitter=Transmitter(
                id=0,
                point=p,
                power=1_000,
                erp=1_000.0,
                antenna_height=0.0,
                antenna_diameter=diameter,
                antenna_gain=calculate_antenna_gain(
                    diameter,
                    frequency_to_wavelength(frequency),
                    efficiency_value=antenna_efficiency,
                ),
                frequency=frequency,
                pulse_width=1.0,
                polarization=Polarization.VERTICAL,
                bandwidth=100,
                antenna_efficiency_value=antenna_efficiency,
            ),
            receiver=Receiver(
                id=0,
                point=p,
                antenna_height=0,
                diameter=2.0,
                cpi_pulses=1,
                pfa=1e-6,
                min_elevation=-20.0,
                max_elevation=60.0,
                rotation_time=10,
                bandwidth=100.0,
            ),
            error_model=MonostaticRadarMeasurementModel(),
        )

        target = Target(
            id=0,
            point=Point(
                lat=47.348,
                lon=8.6266,
                alt=1000.0,
            ),
            cross_section_model=ConstantRcsModel(rcs=2.0),
            velocity=CoordinateTransformations.velocity_geodetic_to_cartesian(
                p,
                0.0,
                -250.0,
                0.0,
            ),
        )

        calculated = calculate_doppler_shift(
            radar.receiver,
            target,
            radar.transmitter,
        )
        self.assertAlmostEqual(calculated, correct, delta=TOLERANCE)

        # correct = 226.27107714677888
        # calculated = calculate_doppler_shift(
        #     1000.0,
        #     47.36700085728634,
        #     8.537724304199216,
        #     407.83600886023686,
        #     47.406,
        #     8.3926,
        #     1000.0,
        #     250.0,
        #     0.0,
        #     0.0,
        # )
        # self.assertAlmostEqual(calculated, correct, delta=TOLERANCE)


if __name__ == "__main__":
    unittest.main()
