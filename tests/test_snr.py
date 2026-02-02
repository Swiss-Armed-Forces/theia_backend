import unittest

import scipy.constants as sc

from theia.coordinates import POSITIONS_OF_INTEREST
from theia.data_loading import elevationAt
from theia.snr import calculate_snr
from theia.types import Point, Polarization, Transmitter
from theia.util import from_dB, to_dB


class SnrTest(unittest.TestCase):
    def setUp(self):
        p = Point(
            lat=POSITIONS_OF_INTEREST["CH_CENTER"]["lat"],
            lon=POSITIONS_OF_INTEREST["CH_CENTER"]["lon"],
            alt=elevationAt(
                POSITIONS_OF_INTEREST["CH_CENTER"]["lat"],
                POSITIONS_OF_INTEREST["CH_CENTER"]["lon"],
            ),
        )
        # Properties of an AN/TPS-70.
        # Unless given otherwise, the values are from Wikipedia:
        # https://en.wikipedia.org/wiki/AN/TPS-70
        self.transmitter = Transmitter(
            id=0,
            point=p,
            power=6.2 * 1e3,
            erp=6.2 * 1e3,  # guess
            antenna_height=10,  # guess
            antenna_diameter=10,  # guess?
            frequency=2.95 * 1e3,
            pulse_width=6.5,
            polarization=Polarization.VERTICAL,  # guess?
            bandwidth=100,  # based on specification of 2.9-3.0 GHz
        )

    def test_formula(self):
        G_T = to_dB(1.0)
        G_R = to_dB(1.0)
        rcs = 2.0
        d = 10_000.0
        cpi = 1
        T = 300
        F = to_dB(1.6)
        L = to_dB(2.0)

        wavelength = sc.speed_of_light / (self.transmitter.frequency * 1e6)

        snr = calculate_snr(
            wavelength,
            G_T,
            G_R,
            rcs,
            d,
            self.transmitter.power,
            self.transmitter.bandwidth,
            cpi,
            T,
            F,
            L,
        )

        # Value calculated "by hand".
        snr_true = to_dB(4.8712357411401492846 * 1e-6)

        self.assertAlmostEqual(snr, snr_true, delta=0.005)

        # Test scaling behaviour.
        FACTORS = [0.001, 0.01, 0.1, 10, 100, 1000]
        snr_linear = from_dB(snr)
        for a in FACTORS:
            # Wavelength: Quadratic behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    a * wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    self.transmitter.power,
                    self.transmitter.bandwidth,
                    cpi,
                    T,
                    F,
                    L,
                )
            )
            snr_linear_scaled_expected = snr_linear * a**2
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
                delta=0.005,
            )

            # Transmission gain: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    to_dB(from_dB(G_T) * a),
                    G_R,
                    rcs,
                    d,
                    self.transmitter.power,
                    self.transmitter.bandwidth,
                    cpi,
                    T,
                    F,
                    L,
                )
            )
            snr_linear_scaled_expected = snr_linear * a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Receiver gain: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    to_dB(from_dB(G_R) * a),
                    rcs,
                    d,
                    self.transmitter.power,
                    self.transmitter.bandwidth,
                    cpi,
                    T,
                    F,
                    L,
                )
            )
            snr_linear_scaled_expected = snr_linear * a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # RCS: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    a * rcs,
                    d,
                    self.transmitter.power,
                    self.transmitter.bandwidth,
                    cpi,
                    T,
                    F,
                    L,
                )
            )
            snr_linear_scaled_expected = snr_linear * a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Distance: Power -4 behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    a * d,
                    self.transmitter.power,
                    self.transmitter.bandwidth,
                    cpi,
                    T,
                    F,
                    L,
                )
            )

            snr_linear_scaled_expected = snr_linear / a**4
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Power: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    a * self.transmitter.power,
                    self.transmitter.bandwidth,
                    cpi,
                    T,
                    F,
                    L,
                )
            )

            snr_linear_scaled_expected = snr_linear * a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Bandwidth: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    self.transmitter.power,
                    a * self.transmitter.bandwidth,
                    cpi,
                    T,
                    F,
                    L,
                )
            )

            snr_linear_scaled_expected = snr_linear / a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # CPI: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    self.transmitter.power,
                    self.transmitter.bandwidth,
                    a * cpi,
                    T,
                    F,
                    L,
                )
            )

            snr_linear_scaled_expected = snr_linear * a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Temperature: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    self.transmitter.power,
                    self.transmitter.bandwidth,
                    cpi,
                    a * T,
                    F,
                    L,
                )
            )

            snr_linear_scaled_expected = snr_linear / a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Loss figure: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    self.transmitter.power,
                    self.transmitter.bandwidth,
                    cpi,
                    T,
                    to_dB(a * from_dB(F)),
                    L,
                )
            )

            snr_linear_scaled_expected = snr_linear / a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Losses: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    self.transmitter.power,
                    self.transmitter.bandwidth,
                    cpi,
                    T,
                    F,
                    to_dB(a * from_dB(L)),
                )
            )

            snr_linear_scaled_expected = snr_linear / a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )


if __name__ == "__main__":
    unittest.main()
