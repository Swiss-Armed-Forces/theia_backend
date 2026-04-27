import unittest

import scipy.constants as sc

from theia.coordinates import POSITIONS_OF_INTEREST
from theia.terrain import elevationAt
from theia.snr import calculate_snr
from theia.types import Point, Polarization, Transmitter, calculate_antenna_gain
from theia.util import frequency_to_wavelength, from_dB, to_dB


class SnrTest(unittest.TestCase):
    def setUp(self):
        frequency = 2.95 * 1e3  # MHz
        antenna_efficiency = 0.6
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
            antenna_gain=calculate_antenna_gain(
                10,
                frequency_to_wavelength(frequency),
                efficiency_value=antenna_efficiency,
            ),
            antenna_efficiency_value=antenna_efficiency,
            frequency=frequency,
            pulse_width=6.5,
            polarization=Polarization.VERTICAL,  # guess?
            bandwidth=100,  # based on specification of 2.9-3.0 GHz
        )

    def test_formula(self):
        power = 6.2 * 1e3
        bandwidth = 100
        G_T = to_dB(1.0)
        G_R = to_dB(1.0)
        rcs = 2.0
        d = 10_000.0
        cpi = 1
        T = 300
        L_t = to_dB(2.0)
        L_r = 1.5
        L_a = to_dB(1.0)
        pol = to_dB(1.0)
        F_t2 = to_dB(3.0)
        F_r2 = to_dB(1.0)

        wavelength = sc.speed_of_light / (self.transmitter.frequency * 1e6)

        snr = calculate_snr(
            wavelength,
            G_T,
            G_R,
            rcs,
            d,
            d,
            power,
            bandwidth,
            cpi,
            T,
            L_t,
            L_r,
            L_a,
            pol,
            F_t2,
            F_r2,
        )

        # Value calculated "by hand" according to the SNR formula in the documentation.
        snr_true = 37.924 + 2 * (-9.930) + 3.010 + 0 + 4.771 + 0 - (32.976 - 228.599 + 24.771 + 1.5 + 80 + 4 * 40 + 3.010 + 0)

        self.assertAlmostEqual(float(snr), snr_true, delta=0.005)

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
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
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
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
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
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
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
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
                )
            )
            snr_linear_scaled_expected = snr_linear * a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Distance Tx - target: Power -2 behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    a * d,
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
                )
            )

            snr_linear_scaled_expected = snr_linear / a**2
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Distance Rx - target: Power -2 behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    a * d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
                )
            )

            snr_linear_scaled_expected = snr_linear / a**2
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
                    d,
                    a * power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
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
                    d,
                    power,
                    a * bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
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
                    d,
                    power,
                    bandwidth,
                    a * cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
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
                    d,
                    power,
                    bandwidth,
                    cpi,
                    a * T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
                )
            )

            snr_linear_scaled_expected = snr_linear / a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # L_t: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    to_dB(a * from_dB(L_t)),
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
                )
            )

            snr_linear_scaled_expected = snr_linear / a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # L_r: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    to_dB(a * from_dB(L_r)),
                    L_a,
                    pol,
                    F_t2,
                    F_r2,
                )
            )

            snr_linear_scaled_expected = snr_linear / a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # L_a: Linear behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    to_dB(a * from_dB(L_a)),
                    pol,
                    F_t2,
                    F_r2,
                )
            )

            snr_linear_scaled_expected = snr_linear / a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # polarization factor: Quadratic behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    to_dB(a * from_dB(pol)),
                    F_t2,
                    F_r2,
                )
            )

            snr_linear_scaled_expected = snr_linear * a**2
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Transmitter antenna pattern: Quadratic behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    to_dB(a * from_dB(F_t2)),
                    F_r2,
                )
            )

            snr_linear_scaled_expected = snr_linear * a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )

            # Receiver antenna pattern: Quadratic behaviour.
            snr_linear_scaled = from_dB(
                calculate_snr(
                    wavelength,
                    G_T,
                    G_R,
                    rcs,
                    d,
                    d,
                    power,
                    bandwidth,
                    cpi,
                    T,
                    L_t,
                    L_r,
                    L_a,
                    pol,
                    F_t2,
                    to_dB(a * from_dB(F_r2)),
                )
            )

            snr_linear_scaled_expected = snr_linear * a
            self.assertAlmostEqual(
                snr_linear_scaled,
                snr_linear_scaled_expected,
            )


if __name__ == "__main__":
    unittest.main()
