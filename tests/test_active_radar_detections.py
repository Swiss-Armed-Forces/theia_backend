import math
import unittest
from unittest.mock import patch

import numpy as np
import scipy.constants as sc
from scipy.stats import kstest, norm

from theia.config import ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD, RF_LOSS, SIDC
from theia.coordinates import CoordinateTransformations, EcefToEnuTransformer
from theia.detection.active import (
    FastPd,
    _calculate_probability_of_detection,
    calculate_monostatic_detection,
    calculate_monostatic_snr,
    calculate_probability_of_detection,
)
from theia.terrain import DummyTerrain
from theia.test_data import build_flores_monostatic_radar
from theia.types import (
    ConstantRcsModel,
    MonostaticRadarMeasurementModel,
    MonostaticSensor,
    Point,
    Target,
    Velocity,
)
from theia.util import get_clear_sky_attenuation

RADAR_POINT = Point(lat=46.95, lon=7.45, alt=550.0)
TARGET_RANGE = 20_000.0  # m


# Mocks: Make the target always detectable, so that only the field-of-view
# check decides whether a detection is returned.
def _snr_100dB(*args, **kwargs) -> float:
    return 100.0


def _snr_15dB(*args, **kwargs) -> float:
    """SNR at which the uncertainties are not clipped."""
    return 15.0


def _pd_1(*args, **kwargs) -> float:
    """Always detect, even at moderate SNR."""
    return 1.0


# Low enough minimum angular uncertainty for the Cramér-Rao bound to apply at 15 dB.
_UNCLIPPED_ERROR_MODEL = MonostaticRadarMeasurementModel(
    min_angular_uncertainty=math.radians(0.01)
)


def _doppler_1kHz(*args, **kwargs) -> float:
    return 1_000.0


def _radar(
    min_elevation: float = math.radians(-90.0),
    max_elevation: float = math.radians(90.0),
    min_azimuth: float = math.radians(-180.0),
    max_azimuth: float = math.radians(180.0),
) -> MonostaticSensor:
    """Build a radar whose receiver field-of-view is given in radians."""
    radar = build_flores_monostatic_radar(RADAR_POINT, 0, 1, 2)
    receiver = radar.receiver.model_copy(
        update=dict(
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            min_azimuth=min_azimuth,
            max_azimuth=max_azimuth,
        )
    )
    return radar.model_copy(update=dict(receiver=receiver))


def _target(
    azimuth_deg: float,
    elevation_deg: float,
    target_range: float = TARGET_RANGE,
) -> Target:
    """Place a target at the given azimuth/elevation/range as seen from the radar."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    east = target_range * math.cos(el) * math.sin(az)
    north = target_range * math.cos(el) * math.cos(az)
    up = target_range * math.sin(el)
    ecef = EcefToEnuTransformer(RADAR_POINT).enu_to_ecef((east, north, up))
    lat, lon, alt = CoordinateTransformations.cartesian_to_geodetic(*ecef)
    return Target(
        id=0,
        is_stationary=False,
        sidc=SIDC.UNKNOWN,
        point=Point(lat=lat, lon=lon, alt=alt),
        cross_section_model=ConstantRcsModel(rcs=1.0),
        velocity=Velocity(vx=300.0, vy=0.0, vz=0.0),
    )


@patch("theia.detection.active.calculate_doppler_shift", _doppler_1kHz)
@patch("theia.detection.active.calculate_monostatic_snr", _snr_100dB)
class MonostaticFieldOfViewTest(unittest.TestCase):
    """
    The receiver's field-of-view is stored in radians.

    Regression tests: The field-of-view check used to interpret it in degrees,
    which silently discarded detections e.g. at azimuths beyond 180°.
    """

    def _detect(self, radar: MonostaticSensor, target: Target):
        return calculate_monostatic_detection(
            DummyTerrain(has_los=True),
            radar,
            target,
            np.random.default_rng(seed=0),
        )

    def assertDetected(self, radar, azimuth_deg, elevation_deg):
        detection = self._detect(radar, _target(azimuth_deg, elevation_deg))
        self.assertIsNotNone(
            detection,
            f"Target at az={azimuth_deg}°, el={elevation_deg}° should be detected",
        )
        return detection

    def assertNotDetected(self, radar, azimuth_deg, elevation_deg):
        self.assertIsNone(
            self._detect(radar, _target(azimuth_deg, elevation_deg)),
            f"Target at az={azimuth_deg}°, el={elevation_deg}° should not be detected",
        )

    def test_full_field_of_view_detects_all_azimuths(self):
        radar = _radar()
        for azimuth_deg in range(0, 360, 15):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertDetected(radar, azimuth_deg, 10.0)

    def test_full_field_of_view_detects_all_elevations(self):
        radar = _radar()
        for elevation_deg in (-60.0, -10.0, 0.0, 10.0, 45.0, 80.0):
            with self.subTest(elevation_deg=elevation_deg):
                self.assertDetected(radar, 200.0, elevation_deg)

    def test_detection_angles_match_target_position(self):
        detection = self.assertDetected(_radar(), 270.0, 20.0)
        self.assertAlmostEqual(detection.azimuth_angle, math.radians(270.0), places=3)
        self.assertAlmostEqual(detection.elevation_angle, math.radians(20.0), places=3)

    def test_azimuth_sector_across_north(self):
        # Sector [-90°, +90°], i.e. the northern half.
        radar = _radar(min_azimuth=math.radians(-90), max_azimuth=math.radians(90))
        for azimuth_deg in (0.0, 45.0, 85.0, 275.0, 315.0):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertDetected(radar, azimuth_deg, 10.0)
        for azimuth_deg in (95.0, 180.0, 265.0):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertNotDetected(radar, azimuth_deg, 10.0)

    def test_azimuth_sector_south_west(self):
        # Sector [180°, 270°]
        radar = _radar(min_azimuth=math.radians(180), max_azimuth=math.radians(270))
        for azimuth_deg in (185.0, 225.0, 265.0):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertDetected(radar, azimuth_deg, 10.0)
        for azimuth_deg in (0.0, 90.0, 175.0, 275.0):
            with self.subTest(azimuth_deg=azimuth_deg):
                self.assertNotDetected(radar, azimuth_deg, 10.0)

    def test_elevation_limits(self):
        radar = _radar(
            min_elevation=math.radians(0.0),
            max_elevation=math.radians(30.0),
        )
        for elevation_deg in (1.0, 15.0, 29.0):
            with self.subTest(elevation_deg=elevation_deg):
                self.assertDetected(radar, 90.0, elevation_deg)
        for elevation_deg in (-10.0, 31.0, 60.0):
            with self.subTest(elevation_deg=elevation_deg):
                self.assertNotDetected(radar, 90.0, elevation_deg)


def _wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle [rad] to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class MonostaticDetectionGatesTest(unittest.TestCase):
    """Criteria which reject a target before any measurement is made."""

    def _detect(self, has_los: bool = True, seed: int = 0, **kwargs):
        return calculate_monostatic_detection(
            DummyTerrain(has_los=has_los),
            _radar(),
            _target(90.0, 10.0),
            np.random.default_rng(seed=seed),
            **kwargs,
        )

    @patch("theia.detection.active.calculate_doppler_shift", _doppler_1kHz)
    def test_no_line_of_sight(self):
        self.assertIsNotNone(self._detect(has_los=True))
        self.assertIsNone(self._detect(has_los=False))

    @patch("theia.detection.active.calculate_monostatic_snr", _snr_100dB)
    def test_doppler_threshold(self):
        threshold = 5.0
        for doppler, is_detected in (
            (0.0, False),
            (threshold, False),
            (-threshold, False),
            (1.01 * threshold, True),
            (-1.01 * threshold, True),
        ):
            with (
                self.subTest(doppler=doppler),
                patch(
                    "theia.detection.active.calculate_doppler_shift",
                    return_value=doppler,
                ),
            ):
                self.assertEqual(
                    self._detect(doppler_shift_threshold_hz=threshold) is not None,
                    is_detected,
                )

    @patch("theia.detection.active.calculate_doppler_shift", _doppler_1kHz)
    @patch("theia.detection.active.calculate_monostatic_snr", _snr_100dB)
    def test_probability_of_detection(self):
        for p, is_detected in ((0.0, False), (1.0, True)):
            with patch(
                "theia.detection.active.calculate_probability_of_detection",
                return_value=p,
            ):
                for seed in range(20):
                    with self.subTest(p=p, seed=seed):
                        self.assertEqual(
                            self._detect(seed=seed) is not None, is_detected
                        )


@patch("theia.detection.active.calculate_doppler_shift", _doppler_1kHz)
class MonostaticMeasurementTest(unittest.TestCase):
    """Measured range/elevation/azimuth and their uncertainties."""

    def _detect(self, target, seed=0, error_model=None):
        return calculate_monostatic_detection(
            DummyTerrain(has_los=True),
            _radar(),
            target,
            np.random.default_rng(seed=seed),
            error_model=error_model,
        )

    @patch("theia.detection.active.calculate_monostatic_snr", _snr_100dB)
    def test_perfect_measurement(self):
        target = _target(123.0, 12.0)
        detection = self._detect(target)
        self.assertIsNotNone(detection)
        # Only floating-point round-trip error in the coordinate transformations.
        self.assertAlmostEqual(detection.target_range, TARGET_RANGE, delta=1e-3)
        self.assertAlmostEqual(detection.azimuth_angle, math.radians(123.0), places=6)
        self.assertAlmostEqual(detection.elevation_angle, math.radians(12.0), places=6)
        self.assertEqual(detection.sigma_target_range, 0.0)
        self.assertEqual(detection.sigma_elevation, 0.0)
        self.assertEqual(detection.sigma_azimuth, 0.0)
        self.assertEqual(detection.snr, 100.0)
        self.assertEqual(detection.radar, _radar())
        self.assertEqual(detection.target, target)

    @patch("theia.detection.active.calculate_probability_of_detection", _pd_1)
    @patch("theia.detection.active.calculate_monostatic_snr", _snr_15dB)
    def test_uncertainties_from_error_model(self):
        # Cramér-Rao bound: resolution / (2 * sqrt(SNR)).
        radar = _radar()
        snr = 10 ** (15.0 / 10)
        wavelength = sc.speed_of_light / (radar.transmitter.frequency * 1e6)
        range_resolution = sc.speed_of_light / (2 * radar.receiver.bandwidth * 1e6)
        angular_resolution = wavelength / radar.receiver.diameter
        expected_sigma_range = range_resolution / (2 * math.sqrt(snr))
        expected_sigma_angle = angular_resolution / (2 * math.sqrt(snr))

        error_model = _UNCLIPPED_ERROR_MODEL
        # Otherwise, only the clipping bounds would be tested.
        self.assertTrue(
            error_model.min_range_uncertainty
            < expected_sigma_range
            < error_model.max_range_uncertainty
        )
        self.assertTrue(
            error_model.min_angular_uncertainty
            < expected_sigma_angle
            < error_model.max_angular_uncertainty
        )

        detection = self._detect(_target(90.0, 10.0), error_model=error_model)
        self.assertEqual(detection.snr, 15.0)
        self.assertAlmostEqual(
            detection.sigma_target_range, expected_sigma_range, delta=1e-6
        )
        self.assertAlmostEqual(detection.sigma_elevation, expected_sigma_angle)
        self.assertAlmostEqual(detection.sigma_azimuth, expected_sigma_angle)

    @patch("theia.detection.active.calculate_probability_of_detection", _pd_1)
    @patch("theia.detection.active.calculate_monostatic_snr", _snr_15dB)
    def test_noise_is_normally_distributed(self):
        # Kolmogorov-Smirnov test of the normalized residuals against N(0, 1).
        # Significance level is alpha = 0.05.
        # For n = 500, the test rejects if the maximum CDF distance exceeds
        # D_crit = kstwo.ppf(0.95, 500) ≈ 0.060. Hence, it detects a wrong
        # standard deviation (factor > 1.29 or < 0.78) or a bias > 0.15 sigma,
        # e.g. wrong units or noise applied twice. Smaller errors in the
        # uncertainties are covered by test_uncertainties_from_error_model.
        # The seeds are fixed, so the test is deterministic. With correct code,
        # a set of seeds fails with probability alpha: if an unrelated change of
        # the random draws makes it fail, change the seeds, not the threshold.
        n = 500
        azimuth = math.radians(90.0)
        elevation = math.radians(10.0)
        target = _target(90.0, 10.0)
        detections = [
            self._detect(target, seed=seed, error_model=_UNCLIPPED_ERROR_MODEL)
            for seed in range(n)
        ]
        residuals = {
            "range": [
                (d.target_range - TARGET_RANGE) / d.sigma_target_range
                for d in detections
            ],
            "elevation": [
                (d.elevation_angle - elevation) / d.sigma_elevation for d in detections
            ],
            "azimuth": [
                _wrap_to_pi(d.azimuth_angle - azimuth) / d.sigma_azimuth
                for d in detections
            ],
        }
        for name, r in residuals.items():
            with self.subTest(name):
                self.assertGreaterEqual(kstest(r, "norm").pvalue, 0.05)

    @patch("theia.detection.active.calculate_monostatic_snr", _snr_100dB)
    def test_azimuth_noise_wraps_around_north(self):
        # Regression test: Noisy azimuths used to be clipped to [0, 2*pi)
        # instead of wrapped, which piled up detections at exactly 0 (north).
        #
        # The measured azimuth is X ~ N(μ, σ) with μ = 0.5° (just east of north)
        # and σ = 5°. At 100 dB, the Cramér-Rao bound is far below 5°, so σ is
        # the error model's minimum. With σ ≫ μ, about half of the draws are
        # negative (west of north) and must wrap to just below 360°.
        n = 500
        mu = math.radians(0.5)
        sigma = math.radians(5.0)
        target = _target(math.degrees(mu), 10.0)
        error_model = MonostaticRadarMeasurementModel(min_angular_uncertainty=sigma)
        azimuths = np.array(
            [
                self._detect(target, seed=seed, error_model=error_model).azimuth_angle
                for seed in range(n)
            ]
        )
        self.assertTrue(np.all((0.0 <= azimuths) & (azimuths < 2 * np.pi)))
        # Clipping maps all negative draws (≈46%) to exactly 0. For continuous
        # noise, exactly 0 has probability zero.
        self.assertFalse(np.any(azimuths == 0.0))

        # Mean of the wrapped residuals:
        # - Correct (wrapped): 0, with standard error σ/√n ≈ 0.224°.
        # - Clipped: The measurement is max(X, 0). For X ~ N(μ, σ),
        #   E[max(X, 0)] = μ·Φ(μ/σ) + σ·φ(μ/σ).
        #   With μ/σ = 0.1: Φ = 0.540 and φ = 0.397, so E ≈ 0.27° + 1.99° = 2.26°.
        #   Subtracting the true 0.5° gives a mean residual of about +1.76°
        #   (bias towards east), i.e. 7.8 standard errors.
        # A tolerance of 0.5° = 2.24 standard errors fails correct code with
        # probability 2·(1 - Φ(2.24)) ≈ 2.5% per set of (fixed) seeds.
        mean_residual = np.mean(_wrap_to_pi(azimuths - mu))
        self.assertLess(abs(mean_residual), math.radians(0.5))

        # Fraction of detections west of north (wrapped to (π, 2π)):
        # - Correct: p = P(X < 0) = Φ(-μ/σ) = Φ(-0.1) ≈ 0.460. The observed
        #   fraction is binomial with standard error sqrt(p(1-p)/n) ≈ 0.022.
        # - Clipped: 0.
        # A two-sided tolerance of 3 standard errors (≈ 0.067) fails correct
        # code with probability 2·(1 - Φ(3)) ≈ 0.27% per set of (fixed) seeds.
        # It also catches wrapping bugs that put too many detections west.
        p_west = norm.cdf(-mu / sigma)
        standard_error = math.sqrt(p_west * (1 - p_west) / n)
        self.assertAlmostEqual(
            np.mean(azimuths > np.pi), p_west, delta=3 * standard_error
        )

    @patch("theia.detection.active.calculate_monostatic_snr", _snr_100dB)
    def test_extreme_noise_stays_within_bounds(self):
        # Regression test: Clipping the elevation to exactly +pi/2 used to
        # crash, as MonostaticRadarDetection only accepts [-pi/2, pi/2).
        # The noise is chosen such that each clipping bound is hit often:
        # - Range: σ = 1e6 m ≫ 20 km, so P(range < 0) = Φ(-0.02) ≈ 49%.
        # - Elevation 80° with σ = 180°: P(elevation > 90°) = 1 - Φ(10/180) ≈ 48%.
        # Hence, 200 seeds hit the upper elevation bound with near certainty
        # (the chance of never hitting it is 0.52^200 ≈ 1e-57).
        error_model = MonostaticRadarMeasurementModel(
            min_range_uncertainty=1e6,
            max_range_uncertainty=1e7,
            min_angular_uncertainty=math.radians(180.0),
        )
        for seed in range(200):
            with self.subTest(seed=seed):
                d = self._detect(_target(90.0, 80.0), seed=seed, error_model=error_model)
                self.assertGreaterEqual(d.target_range, 0.0)
                self.assertGreaterEqual(d.elevation_angle, -np.pi / 2)
                self.assertLess(d.elevation_angle, np.pi / 2)
                self.assertGreaterEqual(d.azimuth_angle, 0.0)
                self.assertLess(d.azimuth_angle, 2 * np.pi)


class ProbabilityOfDetectionTest(unittest.TestCase):
    PFAS = (1e-6, 1e-4)

    def test_monotonic_in_snr(self):
        for pfa in self.PFAS:
            with self.subTest(pfa=pfa):
                pd = [
                    calculate_probability_of_detection(snr, pfa)
                    for snr in np.linspace(-40, 30, 701)
                ]
                self.assertTrue(np.all(np.diff(pd) >= 0.0))

    def test_limits(self):
        for pfa in self.PFAS:
            with self.subTest(pfa=pfa):
                # No signal: Only false alarms, i.e. P_d ≈ P_fa.
                # The lookup table clamps below -30 dB, so P_d(-50 dB) = P_d(-30 dB).
                # For small alpha, the Marcum Q function is approximately
                # Q_1(alpha, beta) ≈ exp(-beta²/2)·(1 + alpha²·beta²/4)
                #                  = P_fa·(1 + alpha²·beta²/4),
                # with alpha = 10^((-30 + 3)/20) and beta² = -2·ln(P_fa)
                # (see _calculate_probability_of_detection).
                # This gives P_d/P_fa ≈ 1.014 for P_fa = 1e-6 and ≈ 1.009 for 1e-4.
                self.assertAlmostEqual(
                    calculate_probability_of_detection(-50.0, pfa) / pfa,
                    1.0,
                    delta=0.02,
                )
                # Strong signal: The lookup table saturates at exactly 1 above 20 dB.
                self.assertEqual(calculate_probability_of_detection(50.0, pfa), 1.0)

    def test_lookup_table_matches_exact_calculation(self):
        # The table linearly interpolates on a 0.1 dB grid over [-30, 20] dB.
        # Its error is bounded by h²/8·max|P_d''| with h = 0.1 dB. Measured over
        # the whole grid, it is at most 1.4e-4, so 1e-3 leaves a margin of ~7.
        # The SNRs include both grid ends, grid points (e.g. 10.0) and points
        # between grid points (7.3, 12.55), where the interpolation error peaks.
        for pfa in self.PFAS:
            for snr in (-30.0, -5.0, 0.0, 7.3, 10.0, 12.55, 15.0, 20.0):
                with self.subTest(pfa=pfa, snr=snr):
                    self.assertAlmostEqual(
                        calculate_probability_of_detection(snr, pfa),
                        _calculate_probability_of_detection(snr, pfa),
                        delta=1e-3,
                    )

    def test_lookup_table_outside_of_grid(self):
        table = FastPd.get_table(1e-6)
        self.assertEqual(table(-1000.0), table.pd[0])
        self.assertEqual(table(1000.0), table.pd[-1])


class MonostaticSnrTest(unittest.TestCase):
    def _snr(self, target: Target, has_los: bool = True) -> float:
        return calculate_monostatic_snr(
            DummyTerrain(has_los=has_los),
            _radar(),
            target,
            rcs_model=target.cross_section_model,
            distance_step=30.0,
            doppler_shift_threshold_hz=ACTIVE_RADAR_DOPPLER_SHIFT_THRESHOLD,
            rf_loss=RF_LOSS,
        )

    def test_no_line_of_sight(self):
        self.assertEqual(self._snr(_target(90.0, 10.0), has_los=False), -1000)

    def test_radar_equation_range_dependency(self):
        # SNR ~ R^-4, i.e. 10·log10(10^4) = 40 dB less per decade of range,
        # plus two-way atmospheric attenuation [dB/km] over the extra distance.
        # Everything else in the radar equation is independent of the range,
        # so the relation holds up to floating-point error (measured: ~1e-13 dB).
        r1 = 10_000.0
        r2 = 100_000.0
        snr1 = self._snr(_target(90.0, 10.0, r1))
        snr2 = self._snr(_target(90.0, 10.0, r2))
        attenuation_dB_per_km = get_clear_sky_attenuation(
            _radar().transmitter.frequency
        )
        expected = 40 * math.log10(r2 / r1) + attenuation_dB_per_km * 2 * (
            (r2 - r1) / 1000.0
        )
        self.assertAlmostEqual(snr1 - snr2, expected, delta=1e-9)

    def test_proportional_to_rcs(self):
        # SNR ~ RCS, so 10x the RCS is 10·log10(10) = +10 dB.
        target = _target(90.0, 10.0)
        target_10x = target.model_copy(
            update=dict(cross_section_model=ConstantRcsModel(rcs=10.0))
        )
        self.assertAlmostEqual(
            self._snr(target_10x) - self._snr(target), 10.0, delta=1e-9
        )


if __name__ == "__main__":
    unittest.main()
