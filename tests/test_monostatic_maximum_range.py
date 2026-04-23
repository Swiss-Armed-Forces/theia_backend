import unittest

from theia.radar_equation import calculate_maximum_monostatic_range
from theia.types import (
    MonostaticRadarMeasurementModel,
    MonostaticSensor,
    Point,
    Polarization,
    Receiver,
    Transmitter,
    calculate_antenna_gain,
)
from theia.util import frequency_to_wavelength


class MonostaticMaxRangeTest(unittest.TestCase):
    def test_compare_against_openburst(self):
        """Compare against reference implementation openBURST."""
        p = Point(
            lat=47.36700085728634,
            lon=8.537724304199216,
            alt=407.83600886023686,
        )
        ah = 10.0
        bandwidth = 1
        diameter = 2.0
        frequency = 1000.0
        antenna_efficiency = 0.6
        radar = MonostaticSensor(
            id=0,
            transmitter=Transmitter(
                id=585,
                point=p,
                power=20000,
                erp=800,
                antenna_height=ah,
                antenna_diameter=diameter,
                antenna_gain=calculate_antenna_gain(
                    diameter,
                    frequency_to_wavelength(frequency),
                    antenna_efficiency,
                ),
                frequency=frequency,
                pulse_width=1,
                bandwidth=bandwidth,
                polarization=Polarization.HORIZONTAL,
            ),
            receiver=Receiver(
                id=585,
                point=p,
                antenna_height=ah,
                diameter=2.0,
                cpi_pulses=1,
                pfa=1e-6,
                min_elevation=-20.0,
                max_elevation=60.0,
                rotation_time=10.0,
                bandwidth=bandwidth,
            ),
            error_model=MonostaticRadarMeasurementModel(),
        )

        target_cross_section: float = 2.0
        r_calculated = calculate_maximum_monostatic_range(
            radar,
            target_cross_section,
        )

        r_true = 16500.001  # Resolution of openBURST range calc: 1000m

        self.assertAlmostEqual(r_calculated, r_true, delta=1000.0)


if __name__ == "__main__":
    unittest.main()
