import datetime

import numpy as np
import pydantic
import scipy.constants as sc

import theia.config
from theia.coordinates import calculate_azimuth_angle, calculate_elevation_angle
from theia.distance import (
    get_2d_distance_between_locs_heights,
    get_bistatic_range,
)
from theia.doppler import calculate_doppler_shift
from theia.grids import LatLonHeightGrid
from theia.line_of_sight import has_line_of_sight
from theia.snr import calculate_snr
from theia.types import (
    ConstantRcsModel,
    PclDetection,
    PclSensor,
    Point,
    Receiver,
    Target,
    Transmitter,
    Velocity,
)
from theia.util import get_clear_sky_attenuation


class PclDetector(pydantic.BaseModel):
    snr_threshold: float = theia.config.SNR_THRESHOLD_PCL
    """Signal-to-noise threshold [dB]"""
    doppler_threshold: float = theia.config.DOPPLER_SHIFT_THRESHOLD_PCL
    """Doppler threshold [Hz]"""
    delay_threshold: float = theia.config.DELAY_THRESHOLD_PCL
    """
    Delay threshold [us].

    If the delay
    (signal propagation time deviation from straight line between Tx and Rx)
    is smaller than this value, the geometry is considered to fall into the
    forward scattering domain and a ValueError is raised.
    """
    distance_step: float = 30.0
    """Step width [m] for the line-of-sight check"""

    def _calculate_bistatic_range_doppler(
        self,
        rx: Receiver,
        tx: Transmitter,
        target: Target,
    ) -> tuple[float, float, float]:
        """
        Returns
        -------
        baseline_range: float
            Baseline range (line-of-sight between tx-rx) [km]
        bistatic_range: float
            Bistatic range [km]
        doppler: float
            Bistatic Doppler shift [Hz]
        """
        # Get bistatic range.
        tx_pos = [tx.lat, tx.lon, tx.alt + tx.antenna_height]
        rx_pos = [rx.lat, rx.lon, rx.alt + rx.antenna_height]
        tgt_pos = [target.lat, target.lon, target.alt]
        (bistatic_range_km, tgt_rx_range, tgt_tx_range, baseline_range) = (
            get_bistatic_range(tx_pos, rx_pos, tgt_pos)
        )

        # Now check bistatic Doppler and return if Doppler shift too low.
        doppler = calculate_doppler_shift(rx, target, tx)  # [Hz]

        return float(baseline_range), float(bistatic_range_km), float(doppler)

    def calculate_raw_measurement(
        self, rx: Receiver, tx: Transmitter, tgt: Target
    ) -> tuple[float, float, float]:
        """
        Returns
        -------
        snr_over_rcs: float
            Signal-to-noise-ratio divided by RCS [dB]
        bistatic_range: float
            Bistatic range [m]
        doppler: float
            Bistatic Doppler shift [Hz]

        Raises
        ------
        ValueError
            If the geometry is not in the bistatic regime (delay too small)
        ValueError
            If there is now direct line-of-sight Tx - Target - Rx
        """
        # has_los = has_line_of_sight(
        #     tx.point, tgt.point, self.distance_step
        # ) and has_line_of_sight(rx.point, tgt.point, self.distance_step)

        # if not has_los:
        #     raise ValueError("No line of sight Tx - Target - Rx")

        baseline_range, bistatic_range_km, doppler = (
            self._calculate_bistatic_range_doppler(
                rx,
                tx,
                tgt,
            )
        )

        # Check whether we are in the bistatic regime. Otherwise raise an exception.
        # dist_delay_limit [m], delay_thresh [us], baseline_range in [km] * 1000 in [m]
        dist_delay_limit = self.delay_threshold * sc.speed_of_light / 1e6 + (
            baseline_range * 1000
        )

        r_r = (
            get_2d_distance_between_locs_heights(
                rx.lat,
                rx.lon,
                rx.alt + rx.antenna_height,
                tgt.lat,
                tgt.lon,
                tgt.alt,
            )
            * 1000.0
        )  # distance in meters

        r_t = (
            get_2d_distance_between_locs_heights(
                tx.lat,
                tx.lon,
                tx.alt + tx.antenna_height,
                tgt.lat,
                tgt.lon,
                tgt.alt,
            )
            * 1000.0
        )  # distance in meters

        # print(f"r_r = {r_r:.3f} m")
        # print(f"r_t = {r_t:.3f} m")

        if r_r + r_t < dist_delay_limit:  # checks if delay threshold is valid
            raise ValueError(
                "Delay is too small; we are in the forward scattering regime"
            )

        # print(f"Tx wavelength      = {sc.speed_of_light / (tx.frequency * 1e6)} m")
        # print(f"Tx frequency       = {tx.frequency} MHz")
        # print(f"Tx antenna gain    = {tx.antenna_gain:.1f} dB")
        # print(f"Rx antenna gain    = {rx.antenna_gain(tx.frequency):.1f} dB")
        # print(f"Distance Tx-target = {r_t:.1f} m")
        # print(f"Distance Rx-target = {r_r:.1f} m")
        # print(f"Transmission Power = {tx.power:.1f} W")
        # print(f"Bandwidth          = {rx.bandwidth:.1f} MHz")
        # print(
        #     f"Tx Pattern propagation factor = {calculate_antenna_pattern(tx, tgt.point):.3f} dB"
        # )
        # print(
        #     f"Rx Pattern propagation factor = {calculate_antenna_pattern(rx, tgt.point):.3f} dB"
        # )
        # print(f"L_t                = {get_clear_sky_attenuation(tx.frequency) * (r_r + r_t) / 1000.0:.1f} dB")
        # print(f"T_N                = {rx.noise_temperature:.1f} K")

        snr = calculate_snr(
            wavelength=sc.speed_of_light / (tx.frequency * 1e6),
            antenna_gain_transmitter=tx.antenna_gain,
            antenna_gain_receiver=rx.antenna_gain(tx.frequency),
            radar_cross_section=1.0,
            distance_transmitter_target=r_t,
            distance_receiver_target=r_r,
            transmission_power=tx.power,
            bandwidth=rx.bandwidth,
            cpi_pulses=rx.cpi_pulses,
            equivalent_temperature=rx.noise_temperature,
            L_t=0.0,
            L_r=rx.noise_figure,
            L_a=get_clear_sky_attenuation(tx.frequency) * (r_r + r_t) / 1000.0,
            polarization_factor=0,
            pattern_propagation_factor_transmitter=calculate_antenna_pattern(
                tx, tgt.point
            ),
            pattern_propagation_factor_receiver=calculate_antenna_pattern(
                rx, tgt.point
            ),
            is_one_way=False,
        )

        return snr, bistatic_range_km * 1000.0, doppler

    def minimum_detectable_rcs_vector(
        self, rx: Receiver, tx: Transmitter, points: np.ndarray
    ) -> np.ndarray:
        """
        Calculate minimum detectable RCS for the given PCL sensor at multiple positions.

        Parameters
        ----------
        rx: Receiver
            Receiver
        tx: Transmitter
            Transmitter
        points: np.ndarray
            Geodetic positions at which to evaluate the minimum detectable RCS.
            Shape: (N, 3)
        """
        results = np.empty(points.shape[0], dtype=np.float64)
        results.fill(np.nan)
        for i, point in enumerate(points):
            tgt = Target(
                id=-1,
                is_stationary=True,
                point=Point(lat=point[0], lon=point[1], alt=point[2]),
                cross_section_model=ConstantRcsModel(rcs=1.0),
                velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
            )
            try:
                snr, _, _ = self.calculate_raw_measurement(rx, tx, tgt)
            except ValueError:
                snr = np.nan
            results[i] = calculate_minimum_detectable_rcs(
                snr,
                self.snr_threshold,
            )
        return results

    def minimum_detectable_rcs_grid(
        self,
        rx: Receiver,
        tx: Transmitter,
        grid: LatLonHeightGrid,
    ) -> np.ndarray:
        """
        Calculate minimum detectable RCS for the given PCL sensor at multiple positions.

        Parameters
        ----------
        rx: Receiver
            Receiver
        tx: Transmitter
            Transmitter
        grid: LatLonHeightGrid
            Grid on which to calculate the minimum detectable RCS
        """
        result_vector = self.minimum_detectable_rcs_vector(rx, tx, grid.points)
        return result_vector.reshape(grid.n_points)

    def calculate_pcl_detection(
        self,
        rng: np.random.Generator,
        sensor: PclSensor,
        tgt: Target,
    ) -> PclDetection | None:
        """
        Calculate whether the given geometry leads to a detection using Passive
        Coherent Location (PCL) radar.

        Parameters
        ----------
        rng: np.random.Generator
            Pseudo random number generator
        sensor: PclSensor
            PCL sensor.
        tgt: Target
            Target.

        Raises
        ------
        ValueError
            If the setup is not in the bistatic regime (e. g. the forward scattering)

        Returns
        -------
        detection: PassiveRadarDetection | None
            the detection that was made or None if no detection takes place.
        """
        assert tgt.alt > 0
        try:
            snr, bistatic_range, doppler = self.calculate_raw_measurement(
                sensor.receiver, sensor.transmitter, tgt
            )
        except ValueError:
            # We ignore the forward scattering case.
            return None

        if abs(doppler) < self.doppler_threshold:
            return None

        min_detectable_rcs = calculate_minimum_detectable_rcs(
            snr,
            self.snr_threshold,
        )
        assert min_detectable_rcs > 0.0

        # Doppler shift was good enough if we reached this far, see above.
        # So just check the rcs_thresholds.
        if min_detectable_rcs <= tgt.cross_section_model(
            transmitter=sensor.transmitter,
            receiver=sensor.receiver,
            target=tgt,
        ):
            # Add uncertainty.
            sigma_bistatic_range = sensor.error_model.sigma_bistatic_range(
                snr,
                sensor,
            )
            sigma_doppler = sensor.error_model.sigma_doppler_shift()
            error_bistatic_range = rng.normal(0.0, sigma_bistatic_range)
            error_doppler = rng.normal(0.0, sigma_doppler)
            bistatic_range = np.clip(
                bistatic_range + error_bistatic_range,
                0,
                np.inf,
            )
            doppler_shift = np.clip(
                doppler + error_doppler,
                0,
                np.inf,
            )
            return PclDetection(
                detection_id=-1,
                time=datetime.datetime.fromtimestamp(0),
                sensor=sensor,
                target=tgt,
                bistatic_range=bistatic_range,
                doppler_shift=doppler_shift,
                sigma_bistatic_range=np.abs(error_bistatic_range),
                sigma_doppler_shift=np.abs(error_doppler),
            )
        else:
            return None


def calculate_minimum_detectable_rcs(
    snr_over_rcs: float,
    snr_threshold: float,
) -> float:
    r"""
    Calculate the minimum radar cross section (RCS) that can be detected for
    the given geometry according to an SNR threshold test.

    Parameters
    ----------
    snr: float
        Signal-to-noise ratio [dB] divided by the radar cross section [m^2]
        for the given geometry.
    snr_threshold: float
        Minimum detectable SNR threshold [dB].
    
    Returns
    -------
    min_detectable_rcs: float
        Minimum detectable radar cross section [m^2] corresponding to
        the given SNR and SNR threshold.

    Notes
    -----
    The principle is described in the unpublished master thesis with the
    title "Passive Radar - From Quality Criteria to Coverage Optimization".
    Since the bistatic RCS :math:`\sigma_B` is unknown, the original expression
    for the SNR is reformulated and only the term :math:`SNR / \sigma_B`,
    which can be computed, is used.

    The derivation of the minimum detectable RCS is straightforward

    .. math::

       \begin{aligned}
       SNR &\geq SNR_{threshold}\\
       \frac{SNR}{\sigma_B} &\geq \frac{SNR_{threshold}}{\sigma_B}\\
       \sigma_B &\geq \frac{SNR_{threshold}}{SNR_{\setminus \sigma_B}} =: \sigma_{B, min}
       \end{aligned}.

    """
    rcs = 10.0 ** ((snr_threshold - snr_over_rcs) / 10.0)
    return float(rcs)


def calculate_antenna_pattern(
    transmitter_or_receiver: Transmitter | Receiver,
    point_of_interest: Point,
) -> float:
    """
    Calculate antenna pattern attenuation [dB] for the given geometry.
    """
    p = Point(
        lat=transmitter_or_receiver.lat,
        lon=transmitter_or_receiver.lon,
        alt=transmitter_or_receiver.alt + transmitter_or_receiver.antenna_height,
    )
    theta_bearing = calculate_azimuth_angle(
        p_observer=p,
        p_target=point_of_interest,
    )
    theta_vert = calculate_elevation_angle(
        p_observer=p,
        p_target=point_of_interest,
    )

    horiz_att = (
        transmitter_or_receiver.horizontal_attenuation(theta_bearing)
        if transmitter_or_receiver.horizontal_attenuation is not None
        else 0.0
    )
    vert_att = (
        transmitter_or_receiver.vertical_attenuation(theta_vert)
        if transmitter_or_receiver.vertical_attenuation is not None
        else 0.0
    )

    return -horiz_att - vert_att


def pcl_track_init_update_masks(
    detector: PclDetector,
    sensors: list[PclSensor],
    grid: LatLonHeightGrid,
    rcs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate binary masks for PCL track init (at least 3 sensors detect) and
    track update (at least 1 sensor detects).

    Parameters
    ----------
    detector: PclDetector
        Detector model to use
    sensors: list[PclSensor]
        PCL Sensors
    grid: LatLonHeightGrid
        Grid on which to calculate detectability
    rcs: float
        Radar cross section of the target to be detected.

    Returns
    -------
    track_init_mask: np.ndarray
        Binary mask indicating where on the grid a track can be initialized using
        only the PCL sensors
    track_update_mask: np.ndarray
        Binary mask indicating where on the grid a track can be updated using
        only the PCL sensors
    """
    min_detectable_rcs_grids = [
        detector.minimum_detectable_rcs_grid(
            sensor.receiver,
            sensor.transmitter,
            grid,
        )
        for sensor in sensors
    ]

    detection_grids = [rcs_grid <= rcs for rcs_grid in min_detectable_rcs_grids]
    n_detections_grid = sum(detection_grids)
    track_update_mask = np.logical_and(0 < n_detections_grid, n_detections_grid < 3)
    track_init_mask = 3 <= n_detections_grid

    return track_init_mask, track_update_mask
