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


class ActiveRadarDetection(pydantic.BaseModel):
    detection_id: int
    time: datetime.datetime
    """Date and time at which the detection takes place."""
    radar: Radar
    target: Target
    target_range: float
    """Line-of-sight distance between radar and target [m]"""
    elevation_angle: float
    """Elevation angle between radar (observer) and target in [-pi/2, pi/2) [rad]"""
    azimuth_angle: float
    """Azimuth between radar (observer) and target measured in [0, 2pi) from north [rad]"""

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
    def __init__(self, rcs: float):
        """
        Parameters
        ----------
        rcs: float
            Radar cross section [m^2] to be used for all geometries
        """
        self._rcs = rcs

    def __call__(
        self,
        transmitter: Transmitter,
        receiver: Receiver,
        target: Target,
    ) -> float:
        return self._rcs


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
    # TODO:
    # Add tracks for enemy targets!


class ReceiverController(abc.ABC):
    @abc.abstractmethod
    def get_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        raise NotImplementedError()


class TransmitterController(abc.ABC):
    @abc.abstractmethod
    def get_transmitters(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Transmitter]:
        raise NotImplementedError()


class TargetController(abc.ABC):
    @abc.abstractmethod
    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Target]:
        raise NotImplementedError()


class Controller(ReceiverController, TransmitterController, TargetController):
    def __init__(
        self,
        transmitter_controller: TransmitterController,
        receiver_controller: ReceiverController,
        target_controller: TargetController,
    ):
        self._transmitter_controller = transmitter_controller
        self._receiver_controller = receiver_controller
        self._target_controller = target_controller

    def get_receivers(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Receiver]:
        return self._receiver_controller.get_receivers(situational_picture, dt)

    def get_transmitters(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Transmitter]:
        return self._transmitter_controller.get_transmitters(situational_picture, dt)

    def get_targets(
        self,
        situational_picture: SituationalPicture,
        dt: datetime.timedelta,
    ) -> list[Target]:
        return self._target_controller.get_targets(situational_picture, dt)


class Snapshot(pydantic.BaseModel):
    time: datetime.datetime
    blue_transmitters: list[Transmitter]
    blue_receivers: list[Receiver]
    blue_targets: list[Target]
    red_transmitters: list[Transmitter]
    red_receivers: list[Receiver]
    red_targets: list[Target]
