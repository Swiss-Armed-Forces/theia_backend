from __future__ import annotations
import abc
import datetime
import enum
from typing import Iterable, Optional, Self
from matplotlib import pyplot as plt
import numpy as np
import pydantic
from scipy.interpolate import CubicSpline
import scipy.constants as sc
import shapely

from theia.util import from_dB


def calculate_antenna_gain(
    antenna_diameter: float,
    wavelength: float,
    efficiency_value: float = 0.6,
) -> float:
    r"""
    Calculate antenna gain [dB].

    Parameters
    ----------
    antenna_diameter: float
        Antenna diameter [m].
    wavelength: float
        Signal wavelength [m].
    efficiency_value: float, default 0.6
        Demensionless value in [0, 1] that indicates the fraction of the signal
        power that is received by the antenna.

    Notes
    -----
    The formula implemented is Equ. 2.49 in Skolnik 1980

    .. math::

       G = \rho \frac{4 \pi A}{\lambda^2},

       where :math:`\rho` denotes the antenna efficiency value, :math:`A` the
       aperture area of the antenna and :math:`\lambda` the signal wavelength.

    References
    ----------
    Skolnik, M. I. (1980). Introduction to Radar Systems (2nd ed.). McGraw-Hill.
    """
    antenna_area = np.pi * antenna_diameter * antenna_diameter / 4
    return 10 * np.log10(
        efficiency_value * 4 * np.pi * antenna_area / (wavelength * wavelength)
    )


class RadioClimate(enum.Enum):
    EQUATORIAL = 1
    CONTINENTAL_SUBTROPICAL = 2
    MARITIME_TROPICAL = 3
    DESERT = 4
    CONTINENTAL_TEMPERATE = 5
    MARITIME_TEMPERATE_OVER_LAND = 6
    MARITIME_TEMPERATE_OVER_SEA = 7


class Polarization(enum.Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Point(pydantic.BaseModel):
    lat: float
    """Latitude [decimal °]"""
    lon: float
    """Longitude [decimal °]"""
    alt: float
    """Altitude (meters above sea level) [m]"""

    def as_tuple(self):
        return (self.lat, self.lon, self.alt)


class Velocity(pydantic.BaseModel):
    vx: float
    """Velocity in Cartesian x-direction [m/s]"""
    vy: float
    """Velocity in Cartesian y-direction [m/s]"""
    vz: float
    """Velocity in Cartesian z-direction [m/s]"""

    @property
    def speed(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def as_tuple(self) -> tuple[float, float, float]:
        return [self.vx, self.vy, self.vz]


class AttenuationModel(pydantic.BaseModel):
    attenuation_table_angles: list[float]
    attenuation_table_values: list[float]
    polarization: Polarization

    @staticmethod
    def _vertical_attenuation_half_wave_dipole(theta):
        """
        Calculate the vertical attenuation of a half-wave dipole antenna.

        Parameters
        -----------
        theta: np.typing.ArrayLike
            Elevation angle [rad] (0rad == horizontal plane)

        Returns
        -------
            Vertical attenuation at the given elevation angle [dB]
        """
        if abs(abs(theta) - np.pi / 2) < 1e-5:
            attenuation_factor_db = 100
        else:
            rad_power_factor = np.square(
                np.cos(np.pi / 2 * np.cos(np.pi / 2 + theta))
                / np.sin(np.pi / 2 + theta)
            )
            attenuation_factor_db = -10.0 * np.log10(rad_power_factor)

        return attenuation_factor_db

    def __call__(self, angles: np.typing.ArrayLike) -> float:
        """
        Interpolate the attenuation table linearly at the given angle.

        Parameters
        ----------
        angles: np.typing.ArrayLike
            Angles in [0, 2 pi] at which to evaluate the attenuation table [rad]

        Returns
        -------
        attenuation: np.typing.ArrayLike
            Attenuation values at the given angles [dB];
            has same shape as ``angles``
        """
        if len(self.attenuation_table_angles) == 0:
            if self.polarization == Polarization.VERTICAL:
                return self._vertical_attenuation_half_wave_dipole(angles)
            else:
                return 0.0
        else:
            return np.interp(
                angles,
                self.attenuation_table_angles,
                self.attenuation_table_values,
                left=np.nan,
                right=np.nan,
            )


class Transmitter(pydantic.BaseModel):
    id: int
    """Unique ID"""
    point: Point
    """Coordinates"""
    power: float
    """Power [W]"""
    erp: float
    """Effective radiated power [W]"""
    antenna_height: float
    """Antenna height [m]"""
    antenna_diameter: float
    """Antenna diameter [m]"""
    frequency: float
    """Signal frequency [MHz]"""
    pulse_width: float
    """Pulse width [us]"""
    polarization: Polarization
    """Signal polarization. Only important for PCL."""
    bandwidth: float
    """Noise band width [MHz]"""
    max_coherent_integration_time: float = 0.5
    """
    maximum coherent integration time in [s]
    (use appropriate values for different signals)
    """
    antenna_efficiency_value: float = 0.6
    """
    Fraction in [0, 1] of the transmission power radiated by the antenna.
    """
    vertical_attenuation: Optional[AttenuationModel] = None
    """
    Interpolate the attenuation diagram for elevation angles in [-pi/2, pi/2]
    given in [rad].
    """
    horizontal_attenuation: Optional[AttenuationModel] = None
    """
    Interpolate the attenuation diagram for azimuth angles in [0, 2pi]
    given in [rad].
    """

    @property
    def lat(self) -> float:
        """Latitude [decimal °]"""
        return self.point.lat

    @property
    def lon(self) -> float:
        """Longitude [decimal °]"""
        return self.point.lon

    @property
    def alt(self) -> float:
        """Altitude (meters above sea level) [m]"""
        return self.point.alt

    @property
    def processing_gain(self) -> float:
        """Processing gain [dB]"""
        return 10 * np.log10(self.max_coherent_integration_time * self.bandwidth * 1e6)

    @property
    def antenna_gain(self) -> float:
        """Antenna gain [dB]"""
        return calculate_antenna_gain(
            self.antenna_diameter,
            sc.speed_of_light / (self.frequency * 1e6),
            efficiency_value=self.antenna_efficiency_value,
        )

    @property
    def pulse_compression_gain(self) -> float:
        """Pulse compression gain [dB]"""
        return 10 * np.log10(self.pulse_width * self.bandwidth)


class Receiver(pydantic.BaseModel):
    id: int
    """Unique ID"""
    point: Point
    """Coordinates"""
    antenna_height: float
    """Antenna height [m]"""
    diameter: float
    """Antenna diameter [m]"""
    cpi_pulses: float
    pfa: float
    """Probability of false alarm (in [0, 1])"""
    min_elevation: float
    """Minimum elevation [°]"""
    max_elevation: float
    """Maximum elevation [°]"""
    rotation_time: float
    """Rotation time [s]"""
    bandwidth: float
    """Noise bandwidth of the receiver's predetection filter [MHz]"""
    gain: float = 0
    """Antenna gain [dBi]"""
    losses: float = 0
    """losses from antenna to receiver input [dB]"""
    noise_temperature: float = 300.0
    """Receiving system noise temperature [K]"""
    noise_figure: float = 1.9
    """Receiver LNA noise figure [dB]  (not known for specific radars, best guess)"""
    antenna_efficiency_value: float = 0.6
    """
    Fraction in [0, 1] of the transmission power radiated by the antenna.
    """
    vertical_attenuation: Optional[AttenuationModel] = None
    """
    Interpolate the attenuation diagram for elevation angles in [-pi/2, pi/2]
    given in [rad].
    """
    horizontal_attenuation: Optional[AttenuationModel] = None
    """
    Interpolate the attenuation diagram for azimuth angles in [0, 2pi]
    given in [rad].
    """

    @property
    def lat(self) -> float:
        """Latitude [decimal °]"""
        return self.point.lat

    @property
    def lon(self) -> float:
        """Longitude [decimal °]"""
        return self.point.lon

    @property
    def alt(self) -> float:
        """Altitude (meters above sea level) [m]"""
        return self.point.alt

    def antenna_gain(self, frequency: float) -> float:
        """
        Calculate antenna gain [dB].

        Parameters
        ----------
        frequency: float
            Signal frequency [MHz]
        """
        return calculate_antenna_gain(
            self.diameter,
            sc.speed_of_light / (frequency * 1e6),
            efficiency_value=self.antenna_efficiency_value,
        )

    @property
    def coherent_integration_gain(self) -> float:
        """Coherent integration gain [dB]."""
        return 10 * np.log10(self.cpi_pulses)


class Radar(pydantic.BaseModel):
    transmitter: Transmitter
    receiver: Receiver


class Target(pydantic.BaseModel):
    id: int
    """Unique identifier"""
    point: Point
    """Coordinates"""
    cross_section_model: ConstantRcsModel
    velocity: Velocity

    @property
    def lat(self) -> float:
        return self.point.lat

    @property
    def lon(self) -> float:
        return self.point.lon

    @property
    def alt(self) -> float:
        return self.point.alt

    def almostEqual(self, other: Target, velocity_tol=0.01) -> bool:
        assert type(other) is Target
        return (
            self.id == other.id
            and np.isclose(self.point.as_tuple(), other.point.as_tuple()).all()
            and self.cross_section_model == other.cross_section_model
            and np.isclose(
                self.velocity.as_tuple(), other.velocity.as_tuple(), atol=0.01
            ).all()
        )


class Trajectory(pydantic.BaseModel):
    target_id: int
    times: list[datetime.datetime]
    """Ordered list of times at which the trajectory's waypoints are defined."""
    lats: list[float]
    """Latitude coordinates [°]"""
    lons: list[float]
    """Longitude coordinates [°]"""
    alts: list[float]
    """Altitude coordinates [m above sea level]"""
    vxs: list[float]
    """Velocity components along Cartesian x-coordinate [m / s]"""
    vys: list[float]
    """Velocity components along Cartesian y-coordinate [m / s]"""
    vzs: list[float]
    """Velocity components along Cartesian z-coordinate [m / s]"""
    cross_section_model: ConstantRcsModel

    _spline: CubicSpline = pydantic.PrivateAttr()

    @pydantic.model_validator(mode="after")
    def check_same_length(self) -> Self:
        if (
            len(self.times) != len(self.lats)
            or len(self.times) != len(self.lons)
            or len(self.times) != len(self.alts)
            or len(self.times) != len(self.vxs)
            or len(self.times) != len(self.vys)
            or len(self.times) != len(self.vzs)
        ):
            raise ValueError("Properties are not of same length")
        return self

    @pydantic.model_validator(mode="after")
    def check_times_ordered(self) -> Self:
        if len(self.times) <= 1:
            return self
        for i in range(1, len(self.times)):
            if self.times[i - 1] > self.times[i]:
                raise ValueError("Time steps are not ordered")
        return self

    @pydantic.model_validator(mode="after")
    def _init_spline(self):
        # runs after field validation, but before the final model is returned
        x = [t.timestamp() for t in self.times]
        y = np.stack(
            [
                self.lats,
                self.lons,
                self.alts,
                self.vxs,
                self.vys,
                self.vzs,
            ],
            axis=1,
        )
        self._spline = CubicSpline(x, y, extrapolate=False)
        return self

    def __call__(self, t: datetime.datetime) -> Target | None:
        y = self._spline(t.timestamp())
        if np.isnan(y).any():
            return None
        lat, lon, alt, vx, vy, vz = y
        return Target(
            id=self.target_id,
            point=Point(
                lat=lat,
                lon=lon,
                alt=alt,
            ),
            cross_section_model=self.cross_section_model,
            velocity=Velocity(vx=vx, vy=vy, vz=vz),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Trajectory):
            return NotImplemented
        return (
            (self.target_id == other.target_id)
            and (self.times == other.times)
            and (self.lats == other.lats)
            and (self.lons == other.lons)
            and (self.alts == other.alts)
            and (self.vxs == other.vxs)
            and (self.vys == other.vys)
            and (self.vzs == other.vzs)
        )

    def plot_velocities(self):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(self.times, self.vxs, label="x")
        ax.plot(self.times, self.vys, label="y")
        ax.plot(self.times, self.vzs, label="z")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True)
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Velocity [m / s]", fontsize=16)
        return fig, ax

    def to_geojson(self) -> shapely.geometry.LineString:
        points = []
        for lat, lon in zip(self.lats, self.lons, strict=True):
            points.append((lon, lat))
        return shapely.geometry.LineString(points)

    @staticmethod
    def from_snapshot(
        target: Target,
        t_min: datetime.datetime,
        t_max: datetime.datetime,
        dt: datetime.timedelta = datetime.timedelta(seconds=10),
    ) -> Trajectory:
        times = []
        t = t_min
        while t <= t_max:
            times.append(t)
            t = t + dt

        return Trajectory(
            target_id=target.id,
            times=times,
            lats=[target.lat for _ in times],
            lons=[target.lon for _ in times],
            alts=[target.alt for _ in times],
            vxs=[target.velocity.vx for _ in times],
            vys=[target.velocity.vy for _ in times],
            vzs=[target.velocity.vz for _ in times],
            cross_section_model=target.cross_section_model,
        )


class TargetSimulator(abc.ABC):
    @abc.abstractmethod
    def get_targets(self, time: datetime.datetime) -> Iterable[Target]:
        """Return an iterator over targets at the given time."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_minimum_time(self) -> datetime.datetime:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_maximum_time(self) -> datetime.datetime:
        raise NotImplementedError()


class RadarSimulator(abc.ABC):
    @abc.abstractmethod
    def get_all_radars(self) -> list[Radar]:
        """Return a list of all radars that might be active at some point."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_radars(self, time: datetime.datetime) -> Iterable[Radar]:
        """Return an iterator over radars at the given time."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_minimum_time(self) -> datetime.datetime:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_maximum_time(self) -> datetime.datetime:
        raise NotImplementedError()


class MonostaticRadarDetection(pydantic.BaseModel):
    detection_id: int
    time: datetime.datetime
    """Date and time at which the detection takes place."""
    radar: Radar
    target: Target
    snr: float
    """Signal-to-noise ratio [dB]"""
    target_range: float
    """Line-of-sight distance between radar and target [m]"""
    elevation_angle: float
    """Elevation angle between radar (observer) and target in [-pi/2, pi/2) [rad]"""
    azimuth_angle: float
    """Azimuth between radar (observer) and target measured in [0, 2pi) from north [rad]"""
    sigma_target_range: float
    """Standard deviation of Gaussian range uncertainty [m]"""
    sigma_elevation: float
    """Standard deviation of Gaussian elevation uncertainty [rad]"""
    sigma_azimuth: float
    """Standard deviation of Gaussian azimuth uncertainty [rad]"""

    @pydantic.model_validator(mode="after")
    def check_range_positivity(self) -> Self:
        if self.target_range < 0:
            raise ValueError(f"Range {self.target_range} < 0")
        return self

    @pydantic.model_validator(mode="after")
    def check_elevation_bounds(self) -> Self:
        if not -np.pi / 2 <= self.elevation_angle < np.pi / 2:
            raise ValueError(f"Elevation {self.elevation_angle} not in [-pi/2, pi/2)")
        return self

    @pydantic.model_validator(mode="after")
    def check_azimuth_bounds(self) -> Self:
        if not 0 <= self.azimuth_angle < 2 * np.pi:
            raise ValueError(f"Azimuth {self.azimuth_angle} not in [0, 2pi)")
        return self


class PassiveRadarDetection(pydantic.BaseModel):
    detection_id: int
    time: datetime.datetime
    """Date and time at which the detection takes place."""
    radar: Radar
    target: Target
    bistatic_range: float
    """Bistatic range [m]."""
    doppler_shift: float
    """Doppler shift [Hz]."""


class PetDetection(pydantic.BaseModel):
    """Representation of a detection from Passive Emitter Tracking."""

    detection_id: int
    time: datetime.datetime
    """Date and time at which the detection takes place."""
    radar: Radar
    target: Target
    azimuth: float
    """Azimuth angle [rad] of the gaze vector towards the transmitter."""
    elevation: float
    """Elevation angle [rad] of the gaze vector towards the transmitter."""


class RcsModel(abc.ABC):
    """Abstract base class for a radar cross section model."""

    @abc.abstractmethod
    def __call__(
        self,
        transmitter: Transmitter,
        receiver: Receiver,
        target: Target,
    ) -> float:
        """
        Calculate the radar cross section for the given transmitter, receiver
        and target geometry.

        Parameter
        ---------
        transmitter: Transmitter
        receiver: Receiver
        target: Target

        Returns
        -------
        float
            Radar cross section [m^2]
        """
        raise NotImplementedError()


class ConstantRcsModel(RcsModel, pydantic.BaseModel):
    rcs: float
    """Radar cross section [m^2] to be used for all geometries"""

    def __call__(
        self,
        transmitter: Transmitter,
        receiver: Receiver,
        target: Target,
    ) -> float:
        return self.rcs


CLUTTER_TARGET = Target(
    id=-2,
    point=Point(lat=0, lon=0, alt=0),
    cross_section_model=ConstantRcsModel(rcs=0),
    velocity=Velocity(vx=0.0, vy=0.0, vz=0.0),
)
"""
Special target representing clutter.
It is needed because a detection needs to be associated to a target.
"""


class MonostaticRadarMeasurementModel(pydantic.BaseModel):
    radar: Radar
    min_range_uncertainty: float = 100.0
    min_angular_uncertainty: float = np.deg2rad(1.0)

    @property
    def range_resolution(self) -> float:
        """Resolution of the detected range [m]"""
        return sc.speed_of_light / (2 * self.radar.receiver.bandwidth * 1e6)

    @property
    def elevation_resolution(self) -> float:
        """Resolution of the detected elevation [rad]"""
        return (
            sc.speed_of_light
            / (self.radar.transmitter.frequency * 1e6)
            / self.radar.receiver.diameter
        )

    @property
    def azimuth_resolution(self) -> float:
        """Resolution of the detected azimuth [rad]"""
        return (
            sc.speed_of_light
            / (self.radar.transmitter.frequency * 1e6)
            / self.radar.receiver.diameter
        )

    def calculate_range_uncertainty(self, snr: float) -> float:
        """
        Calculate range uncertainty [m].

        Parameters
        ----------
        snr: float
            Signal-to-noise ratio of the detection [dB]

        Returns
        -------
        float
            Uncertainty of the range [m]
        """
        cramer_rao_bound = self.range_resolution / (2 * np.sqrt(from_dB(snr)))
        return max(cramer_rao_bound, self.min_range_uncertainty)

    def calculate_elevation_uncertainty(self, snr: float) -> float:
        """
        Calculate elevation uncertainty [rad].

        Parameters
        ----------
        snr: float
            Signal-to-noise ratio of the detection [dB]

        Returns
        -------
        float
            Uncertainty of the elevation [rad]

        Notes
        -----
        This is the Cramér-Rao lower bound.
        """
        cramer_rao_bound = self.elevation_resolution / (2 * np.sqrt(from_dB(snr)))
        return max(cramer_rao_bound, self.min_angular_uncertainty)

    def calculate_azimuth_uncertainty(self, snr: float) -> float:
        """
        Calculate azimuth uncertainty [rad].

        Parameters
        ----------
        snr: float
            Signal-to-noise ratio of the detection [dB]

        Returns
        -------
        float
            Uncertainty of the azimuth [rad]

        Notes
        -----
        This is the Cramér-Rao lower bound.
        """
        cramer_rao_bound = self.azimuth_resolution / (2 * np.sqrt(from_dB(snr)))
        return max(cramer_rao_bound, self.min_angular_uncertainty)

    def sample_clutter(
        self,
        rng: np.random.Generator,
        max_range: float,
    ) -> list[MonostaticRadarDetection]:
        N_range_cells = int(np.ceil(max_range / self.range_resolution))
        N_azimuth_cells = int(np.ceil(2 * np.pi / self.azimuth_resolution))
        N_elevation_cells = int(np.ceil(np.pi / self.elevation_resolution))

        N_cells = N_range_cells * N_azimuth_cells * N_elevation_cells
        N_expected_false_alarms = self.radar.receiver.pfa * N_cells
        N = rng.poisson(N_expected_false_alarms)

        # print(N)

        clutter = []
        for _ in range(N):
            # Two-step sampling:
            # 1. Randomly select a resolution cell in which a false alarm (clutter) occurs.
            # 2. Sample uniformly within the resolution cell.
            range_index = rng.integers(0, N_range_cells)
            azimuth_index = rng.integers(0, N_azimuth_cells)
            elevation_index = rng.integers(0, N_elevation_cells)

            # Approximation:
            # We neglect the curvature of a resolution cell and simply sample
            # elevation, azimuth and range uniformly in each coordinate.
            clutter_range = rng.uniform(
                range_index * self.range_resolution,
                (range_index + 1) * self.range_resolution,
            )
            clutter_azimuth = rng.uniform(
                azimuth_index * self.azimuth_resolution,
                (azimuth_index + 1) * self.azimuth_resolution,
            )
            clutter_elevation = rng.uniform(
                -np.pi / 2.0 + elevation_index * self.elevation_resolution,
                -np.pi / 2.0 + (elevation_index + 1) * self.elevation_resolution,
            )

            # We have to clip the values due to rounding the number of cells up.
            clutter_range = np.clip(clutter_range, a_min=0.0, a_max=max_range)
            clutter_azimuth = np.clip(
                clutter_azimuth,
                a_min=0.0,
                a_max=2 * np.pi - 1e-6,
            )
            clutter_elevation = np.clip(
                clutter_elevation,
                a_min=-np.pi / 2,
                a_max=np.pi / 2 - 1e-6,
            )

            clutter.append(
                MonostaticRadarDetection(
                    detection_id=-1,
                    time=datetime.datetime.fromtimestamp(0),
                    radar=self.radar,
                    target=CLUTTER_TARGET,
                    snr=np.nan,
                    target_range=clutter_range,
                    elevation_angle=clutter_elevation,
                    azimuth_angle=clutter_azimuth,
                    # The noise model is inferred from standard deviation of the
                    # uniform distribution:
                    # sigma_x = |interval| / sqrt(12)
                    sigma_target_range=self.range_resolution / np.sqrt(12.0),
                    sigma_elevation=self.elevation_resolution / np.sqrt(12.0),
                    sigma_azimuth=self.azimuth_resolution / np.sqrt(12.0),
                )
            )
        return clutter


class Situation(pydantic.BaseModel):
    radars: list[Radar]
    targets: list[Target]
    transmitter_labels: dict[int, str] = {}
    """Labels per transmitter ID"""
    receiver_labels: dict[int, str] = {}
    """Labels per receiver ID"""
    target_labels: dict[int, str] = {}
    """Labels per target ID"""


class SituationalPicture(pydantic.BaseModel):
    time: datetime.datetime
    friendly_radars: list[Radar]
    friendly_targets: list[Target]
    enemy_targets: list[Track]


class Controller(abc.ABC):
    @abc.abstractmethod
    def get_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_transmitters(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Transmitter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Target]:
        raise NotImplementedError()


class Snapshot(pydantic.BaseModel):
    time: datetime.datetime
    blue_transmitters: list[Transmitter]
    blue_receivers: list[Receiver]
    blue_targets: list[Target]
    red_transmitters: list[Transmitter]
    red_receivers: list[Receiver]
    red_targets: list[Target]


class Track(pydantic.BaseModel):
    states: list[tuple[datetime.datetime, np.ndarray]]

    _times: list[float] = pydantic.PrivateAttr()
    _y: np.ndarray = pydantic.PrivateAttr()
    _f: CubicSpline = pydantic.PrivateAttr()

    def __init__(self, states: list[tuple[datetime.datetime, np.ndarray]]):
        super().__init__(states=states)

        self._times = []
        self._y = np.empty((len(states), 6), dtype=np.float64)

        for i, (time, state) in enumerate(states):
            self._times.append(time.timestamp())
            self._y[i, :] = state

        self._f = CubicSpline(self._times, self._y)

    def __call__(self, time: datetime) -> np.ndarray:
        return self._f(time.timestamp())

    class Config:
        arbitrary_types_allowed = True


class AbstractTracker(abc.ABC):
    @abc.abstractmethod
    def add_detections(self, detections: list[MonostaticRadarDetection]):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_tracks(self) -> list[Track]:
        raise NotImplementedError()

    # TODO: Include PCL and PET detections...
