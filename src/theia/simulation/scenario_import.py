from __future__ import annotations

import abc
import datetime
import json
import logging
from typing import Annotated, ClassVar, Literal, Optional

import matplotlib
import numpy as np
import pydantic

import theia
from theia.config import (
    BM_COMBAT_RANGE,
    CM_COMBAT_RANGE,
    CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE,
    CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE,
    CM_DEFAULT_SAMPLE_SPACING,
    CM_DEFAULT_TERMINAL_DIVE_ANGLE,
    DRONE_DEFAULT_COMBAT_RANGE,
    GBAD_MIN_TERRAIN_CLEARANCE,
    TERRAIN_HBV_DATA_DIR,
)
from theia.coordinates import POSITIONS_OF_INTEREST, CoordinateTransformations
from theia.cruise_missile import (
    CruiseMissilePath,
    build_terrain_following_path,
    default_min_clearance,
    validate_parameters,
)
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.distance import line_of_sight_distance
from theia.effectors import DirectFireEffector, IndirectFireEffector
from theia.grids import LatLonHeightGrid
from theia.simulation.controllers.action_controller import ActionController
from theia.simulation.controllers.camp_site_controller import CampSiteController
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.critical_infrastructure_controller import (
    CriticalInfrastructureController,
)
from theia.simulation.controllers.fixed_path_kamikaze_drone import FixedPathOneWayDrone
from theia.simulation.controllers.geojson_controller import GeoJsonController
from theia.simulation.controllers.known_target_controller import (
    KnownTargetController,
)
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.shooter_reveal_controller import (
    ShooterRevealController,
)
from theia.simulation.controllers.static_gbad_controller import StaticGbadController
from theia.simulation.controllers.static_gbad_coordinator import StaticGbadCoordinator
from theia.simulation.controllers.switchable_radar import (
    SwitchableMonostaticRadarController,
)
from theia.simulation.damage_model import AbstractDamageModel, UniformDamageModel
from theia.simulation.path_generators import (
    BallisticGenerator,
    ClimbCruiseDiveGenerator,
    PathGenerator,
    TerrainFollowingGenerator,
    build_trajectory,
    convert_to_geodetic,
)
from theia.simulation.simulator import (
    NeverCriterion,
    Simulator,
    TerminationCriterion,
    TimeCriterion,
)
from theia.simulation.theia_logging import (
    CompositeSimulationListener,
    FileLogger,
    SituationalPictureBuffer,
)
from theia.simulation.trackers.pseudo_tracker import PseudoTracker
from theia.terrain import AbstractTerrainModel, FillTerrainModel, SrtmTerrainModel
from theia.terrain_fast_los import FastSrtmModel, HbvTree
from theia.types import (
    AbstractTracker,
    AttenuationModel,
    ConstantRcsModel,
    Controller,
    Entity,
    GeoJSONAnnotation,
    GeoJSONFeature,
    GeoJSONKind,
    GeoJSONMultiPolygon,
    IdProvider,
    MonostaticSensor,
    Party,
    PclSensor,
    Point,
    Polarization,
    TargetCategory,
    TargetInfo,
    TargetInfos,
    Trajectory,
)
from theia.util import to_dB


class TerrainFactory(pydantic.BaseModel):
    terrain_name: str
    fill_value: float | None = None

    _terrain_models: pydantic.ClassVar[dict[str, AbstractTerrainModel]] = {}

    def to_terrain(self) -> AbstractTerrainModel:
        # Avoid loading the same terrain twice.
        if self.terrain_name in self._terrain_models:
            return self._terrain_models[self.terrain_name]

        if self.terrain_name == "SRTM":
            terrain = SrtmTerrainModel()
        elif self.terrain_name.startswith("tree_"):
            tree = HbvTree.load(f"{TERRAIN_HBV_DATA_DIR}/{self.terrain_name}.zip")
            terrain = FastSrtmModel(
                tree=tree,
                srtm_model=SrtmTerrainModel(),
                t_min=30.0,
            )
        else:
            raise ValueError(f"Unknown terrain model {self.terrain_name}")

        if self.fill_value is not None:
            terrain = FillTerrainModel(child=terrain, fill_value=self.fill_value)

        self._terrain_models[self.terrain_name] = terrain
        return terrain


class DirectFireEffectorFactory(pydantic.BaseModel):
    type: Literal["direct"] = "direct"
    id: int
    """Effector ID"""
    name: str
    """Human-readable name for this effector"""
    point: Point
    """Position of the effector."""
    combat_range: float
    """Up to which distance a target can be fought [m]"""
    n_attacks_left: int
    """Number of attacks the effector has left"""
    cadence: float
    """Number of attacks per second"""

    def to_effector(self, terrain: AbstractTerrainModel) -> DirectFireEffector:
        return DirectFireEffector(
            id=self.id,
            name=self.name,
            point=self.point,
            combat_range=self.combat_range,
            n_attacks_left=self.n_attacks_left,
            terrain=terrain,
            cadence=self.cadence,
        )


class IndirectFireEffectorFactory(pydantic.BaseModel):
    type: Literal["indirect"] = "indirect"
    id: int
    """Effector ID"""
    name: str
    """Human-readable name for this effector"""
    point: Point
    """Position of the effector."""
    combat_range: float
    """Up to which distance a target can be fought [m]"""
    n_attacks_left: int
    """Number of attacks the effector has left"""
    cadence: float
    """Number of attacks per second"""
    projectile: DirectFireEffectorFactory
    """
    Template for the launched projectile. Its `id` is never used directly:
    each shot deep-copies this template and mints a fresh id from a separate
    runtime id-space (see Simulator._execute_attacks,
    Entity.DIRECT_FIRE_EFFECTOR), independently of the scenario-level
    Entity.EFFECTOR id-space this factory's own `id` belongs to.
    """
    projectile_speed: float
    """Cruise speed of the launched projectile [m / s]"""
    projectile_max_dist: float
    """Maximum distance the launched projectile can travel [m]"""
    projectile_rcs: ConstantRcsModel
    max_in_flight: int = 1
    """
    Maximum number of this effector's own projectiles allowed to be in
    flight (launched but not yet resolved) at the same time.
    """

    def to_effector(
        self, terrain: AbstractTerrainModel, is_blue: bool
    ) -> IndirectFireEffector:
        return IndirectFireEffector(
            id=self.id,
            name=self.name,
            point=self.point,
            combat_range=self.combat_range,
            n_attacks_left=self.n_attacks_left,
            cadence=self.cadence,
            projectile=self.projectile.to_effector(terrain),
            projectile_speed=self.projectile_speed,
            projectile_max_dist=self.projectile_max_dist,
            projectile_info=TargetInfos.INTERCEPTOR.with_party(
                Party.from_is_blue(is_blue)
            ),
            projectile_rcs=self.projectile_rcs,
            max_in_flight=self.max_in_flight,
        )


class StaticGbadFactory(pydantic.BaseModel):
    target_id: int
    rcs: float
    gbad: DirectFireEffectorFactory | IndirectFireEffectorFactory = pydantic.Field(
        discriminator="type"
    )
    name: str = ""
    geojson_range_altitudes: set[int] = set()
    """
    Altitudes at which the effector range is published as GeoJSON [m above
    sea level]. Only direct-fire GBADs have such a range.
    """

    @pydantic.model_validator(mode="before")
    @classmethod
    def _default_legacy_gbad_type(cls, data):
        # Scenario files saved before indirect fire existed have no "type" on
        # their gbad, which the discriminator needs to pick a factory variant.
        if isinstance(data, dict):
            gbad = data.get("gbad")
            if isinstance(gbad, dict) and "type" not in gbad:
                data = {**data, "gbad": {**gbad, "type": "direct"}}
        return data

    def to_controller(
        self,
        terrain: AbstractTerrainModel,
        is_blue: bool,
    ) -> StaticGbadController:
        self._warn_if_on_or_below_terrain(terrain)
        effector = (
            self.gbad.to_effector(terrain, is_blue)
            if isinstance(self.gbad, IndirectFireEffectorFactory)
            else self.gbad.to_effector(terrain)
        )
        c = StaticGbadController(
            target_id=self.target_id,
            info=TargetInfos.GBAD.with_party(Party.from_is_blue(is_blue)),
            rcs=self.rcs,
            effector=effector,
            geojson_range_altitudes=self.geojson_range_altitudes,
        )
        return c

    def _warn_if_on_or_below_terrain(self, terrain: AbstractTerrainModel):
        # The terrain is only built in ScenarioFactory.to_simulator (loading it
        # e.g. for SRTM is expensive), so this cannot be a pydantic validator.
        p = self.gbad.point
        elevation = terrain.elevationAt(p.lat, p.lon)
        if p.alt - elevation < GBAD_MIN_TERRAIN_CLEARANCE:
            logging.warning(
                f"GBAD '{self.name}' (target ID {self.target_id}) at "
                f"({p.lat:.5f}, {p.lon:.5f}, {p.alt:.1f} m) is "
                f"{p.alt - elevation:+.1f} m above the terrain ({elevation:.1f} m), "
                f"i. e. less than {GBAD_MIN_TERRAIN_CLEARANCE} m: its "
                "line-of-sight checks start at or inside the terrain and are "
                "unreliable."
            )


class FixedPathOneWayDroneFactory(pydantic.BaseModel):
    effector: DirectFireEffectorFactory
    trajectory: Trajectory
    assigned_goal: Point

    def to_controller(
        self,
        terrain: AbstractTerrainModel,
        is_blue: bool,
    ) -> LivingController[FixedPathOneWayDrone]:
        # The party is decided by the side the ORBAT is loaded for, not by the file.
        trajectory = self.trajectory.model_copy(
            update={
                "target_info": self.trajectory.target_info.with_party(
                    Party.from_is_blue(is_blue)
                )
            }
        )
        drone = FixedPathOneWayDrone(
            effector=self.effector.to_effector(terrain),
            trajectory=trajectory,
            assigned_goal=self.assigned_goal,
            terrain=terrain,
        )
        return LivingController(child=drone, target_id=drone.trajectory.target_id)


class AntennaDiagramFactory(pydantic.BaseModel):
    name: Literal["none", "csc_2deg"]

    def to_attenuation_model(self) -> AttenuationModel | None:
        if self.name == "none":
            return None

        if self.name == "csc_2deg":

            def cosecant(x):
                return 1 / (np.sin(x) + 1e-12)

            def antenna_diagram(x, x0):
                return (x <= x0) * 1 + (x > x0) * (cosecant(x) / cosecant(x0)) ** 2

            angles = np.linspace(0, 2 * np.pi, 720)

            return AttenuationModel(
                attenuation_table_angles=angles,
                # Minus sign: We need attenuation values, not gain.
                attenuation_table_values=list(
                    -to_dB(antenna_diagram(angles, np.deg2rad(2)))
                ),
                polarization=Polarization.VERTICAL,
            )

        raise ValueError(f"Unknown antenna diagram {self.name}")


class MonostaticSensorFactory(pydantic.BaseModel):
    target_id: int
    rcs: float
    sensor: MonostaticSensor
    vertical_antenna_diagram: AntennaDiagramFactory = AntennaDiagramFactory(name="none")
    name: str = ""

    def to_controller(
        self, is_blue: bool, coverage: list[GeoJSONAnnotation] | None = None
    ) -> MonostaticRadarController:
        attenuation_model = self.vertical_antenna_diagram.to_attenuation_model()
        sensor = self.sensor.model_copy(deep=True)
        sensor.transmitter.vertical_attenuation = attenuation_model
        sensor.receiver.vertical_attenuation = attenuation_model
        c = MonostaticRadarController(
            target_id=self.target_id,
            radar=sensor,
            is_blue=is_blue,
            rcs_model=ConstantRcsModel(rcs=self.rcs),
            coverage=coverage,
        )
        return LivingController(child=c, target_id=self.target_id)


class CriticalInfrastructureFactory(pydantic.BaseModel):
    target_id: int
    name: str
    point: Point
    rcs: float = 100.0
    """Radar cross section [m^2]"""
    tags: frozenset[str] = frozenset()

    def to_controller(self, is_blue: bool) -> Controller:
        target_info = TargetInfos.CRITICAL_INFRASTRUCTURE.with_party(
            Party.from_is_blue(is_blue)
        ).model_copy(deep=True)
        c = CriticalInfrastructureController(
            target_id=self.target_id,
            name=self.name,
            point=self.point,
            info=target_info,
            rcs=self.rcs,
        )
        return LivingController(child=c, target_id=self.target_id)


class _AmmoTemplateBase(pydantic.BaseModel, abc.ABC):
    """
    Physical settings of a round, shared by all rounds of a camp site.

    Position, launch time and IDs are only filled in at launch, see
    :meth:`to_controller`.

    A subclass must implement :meth:`default_path_generator` and the abstract
    properties, either as a property or as a ``ClassVar`` constant; otherwise
    it cannot be instantiated.
    """

    rcs: float
    """Radar cross section [m^2]"""

    @abc.abstractmethod
    def default_path_generator(self) -> PathGenerator:
        """Path generator used when a launch does not specify one."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def round_name(self) -> str:
        """Name of the round's effector"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def round_speed(self) -> float:
        """Speed passed to the path generator [m/s]"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def round_combat_range(self) -> float:
        """Distance to its goal within which the round attacks [m]"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def round_target_info(self) -> TargetInfo:
        """Target info of the round, without party"""
        raise NotImplementedError()

    def to_controller(
        self,
        target_id: int,
        effector_id: int,
        p_start: Point,
        destination: Point,
        t_start: datetime.datetime,
        path_generator: PathGenerator | None,
        terrain: AbstractTerrainModel,
        is_blue: bool,
    ) -> LivingController[FixedPathOneWayDrone]:
        """
        Launch a round: build its fixed path and its one-shot effector.

        Raises
        ------
        ValueError
            If no path can be built for the given start and destination.
        """
        generator = (
            self.default_path_generator() if path_generator is None else path_generator
        )
        path = generator.generate(
            p_start,
            destination,
            t_start,
            self.round_speed,
            terrain,
        )
        trajectory = Trajectory(
            target_id=target_id,
            target_info=self.round_target_info.with_party(Party.from_is_blue(is_blue)),
            times=path.times,
            lats=path.lats,
            lons=path.lons,
            alts=path.alts,
            cross_section_model=ConstantRcsModel(rcs=self.rcs),
        )
        round_controller = FixedPathOneWayDrone(
            effector=DirectFireEffector(
                id=effector_id,
                name=self.round_name,
                point=p_start,
                combat_range=self.round_combat_range,
                n_attacks_left=1,
                terrain=terrain,
                # One-shot round - no rate-of-fire limit applies.
                cadence=float("inf"),
            ),
            trajectory=trajectory,
            assigned_goal=path.end_point,
            terrain=terrain,
        )
        return LivingController(child=round_controller, target_id=target_id)


class BallisticMissileTemplate(_AmmoTemplateBase):
    type: Literal["ballistic_missile"] = "ballistic_missile"
    category: TargetCategory = TargetCategory.SHORT_RANGE_BALLISTIC_MISSILE

    round_name: ClassVar[str] = "ballistic missile"
    round_speed: ClassVar[float] = 0.0
    """The ballistic path follows from the launch angle and the distance."""
    round_combat_range: ClassVar[float] = BM_COMBAT_RANGE

    def default_path_generator(self) -> PathGenerator:
        return BallisticGenerator()

    @property
    def round_target_info(self) -> TargetInfo:
        return TargetInfos.for_category(self.category)


class CruiseMissileTemplate(_AmmoTemplateBase):
    type: Literal["cruise_missile"] = "cruise_missile"
    speed: float = pydantic.Field(gt=0)
    """Constant speed along the path [m/s]"""
    category: TargetCategory = TargetCategory.CRUISE_MISSILE

    round_name: ClassVar[str] = "cruise missile"
    round_combat_range: ClassVar[float] = CM_COMBAT_RANGE

    def default_path_generator(self) -> PathGenerator:
        return TerrainFollowingGenerator()

    @property
    def round_speed(self) -> float:
        return self.speed

    @property
    def round_target_info(self) -> TargetInfo:
        return TargetInfos.for_category(self.category)


class OneWayDroneTemplate(_AmmoTemplateBase):
    type: Literal["oneway_drone"] = "oneway_drone"
    speed: float = pydantic.Field(gt=0)
    """Constant speed along the path [m/s]"""
    combat_range: float = DRONE_DEFAULT_COMBAT_RANGE
    """Distance to its goal within which the drone attacks [m]"""

    round_name: ClassVar[str] = "one-way drone"

    def default_path_generator(self) -> PathGenerator:
        return ClimbCruiseDiveGenerator()

    @property
    def round_speed(self) -> float:
        return self.speed

    @property
    def round_combat_range(self) -> float:
        return self.combat_range

    @property
    def round_target_info(self) -> TargetInfo:
        return TargetInfos.for_category(TargetCategory.DRONE_CLASS_I)


AmmoTemplate = Annotated[
    BallisticMissileTemplate | CruiseMissileTemplate | OneWayDroneTemplate,
    pydantic.Field(discriminator="type"),
]


class CampSite(pydantic.BaseModel):
    """
    Place where a side keeps counted stock of one ammunition type.

    A launch site (``can_launch=True``) can launch its stock; a depot cannot.
    The site itself is a target: if it is destroyed, its stock is lost.
    """

    id: int
    """Camp site ID, unique within the order of battle"""
    name: str
    point: Point
    target_id: int
    """ID to be used to represent this site as a detectable target"""
    rcs: float
    """Radar cross section of the site [m^2]"""
    n_ammo: int = pydantic.Field(ge=0)
    """Number of rounds in stock at the start"""
    ammo: AmmoTemplate
    can_launch: bool
    seconds_between_launches: float = pydantic.Field(ge=0)
    """Reload time between two launches [s]"""

    def target_info(self, is_blue: bool) -> TargetInfo:
        """Categorical information of the site as a target."""
        return TargetInfo(
            sidc_template=TargetInfos.UNKNOWN.sidc_template,
            tags=frozenset(
                {"camp_site", "launch_site" if self.can_launch else "depot"}
            ),
            party=Party.from_is_blue(is_blue),
        )

    def to_controller(self, is_blue: bool) -> LivingController[CampSiteController]:
        # The controller owns a copy, so this site keeps its initial stock.
        return LivingController(
            child=CampSiteController(
                camp_site=self.model_copy(deep=True), is_blue=is_blue
            ),
            target_id=self.target_id,
        )


class ShooterRevealFactory(pydantic.BaseModel):
    """
    Intel source that reveals the positions of enemy shooters to this side.

    Builds a ``ShooterRevealController`` (see there for the rules).
    """

    type: Literal["shooter_reveal"] = "shooter_reveal"
    seconds_valid: float = pydantic.Field(gt=0)
    """How long a shooter's position stays known after its shot [s]"""

    def to_controller(self, is_blue: bool) -> ShooterRevealController:
        return ShooterRevealController(
            is_blue=is_blue,
            valid_time=datetime.timedelta(seconds=self.seconds_valid),
        )


class KnownTargetFactory(pydantic.BaseModel):
    """
    Lasting intel about a stationary enemy target, e.g. a facility.

    Builds a ``KnownTargetController`` that reports the target for the whole
    simulation. ``point`` and ``target_info`` are this side's belief and are
    used as given (including the party); they need not match a real target.
    The IDs of the reported target are assigned when the simulation starts,
    so the file holds none.
    """

    type: Literal["known_target"] = "known_target"
    name: str = ""
    point: Point
    target_info: TargetInfo

    def to_controller(self, is_blue: bool) -> KnownTargetController:
        return KnownTargetController(point=self.point, target_info=self.target_info)


IntelFactory = Annotated[
    ShooterRevealFactory | KnownTargetFactory,
    pydantic.Field(discriminator="type"),
]
"""Intel sources of a side; add further factories to the union."""


# TODO: Fix this ugly workaround cleanly.
# CampSiteController refers to CampSite, which cannot be imported there at
# runtime (circular import).
CampSiteController.model_rebuild()


class TransportTime(pydantic.BaseModel):
    """
    One-way route for moving stock between two camp sites.

    A return trip needs its own entry.
    """

    source_site_id: int
    target_site_id: int
    seconds_until_leave: float = pydantic.Field(ge=0)
    """Time to break camp at the source site [s]"""
    seconds_underway: float = pydantic.Field(ge=0)
    """Time in transit [s]"""


class MonostaticCoverageCalcSettings(pydantic.BaseModel):
    targetAlt: float
    targetRcs: float
    probabilityThreshold: float
    latRes: float
    lonRes: float


class MonostaticCoverageCalculation(pydantic.BaseModel):
    sensorId: int
    settings: MonostaticCoverageCalcSettings
    coverage: list[GeoJSONFeature]


class PclSensorFactory(pydantic.BaseModel):
    target_id: int
    rcs: float
    sensor: PclSensor

    # TODO: Build a controller from this!


class PclMinDetectableRcsCalcSettings(pydantic.BaseModel):
    grid: LatLonHeightGrid
    snr_threshold: float
    """SNR theshold [dB]"""
    doppler_threshold: float
    """Doppler theshold [Hz]"""
    delay_threshold: float
    """Delay theshold [us]"""


class PclMinDetectableRcsCalculation(pydantic.BaseModel):
    sensor_id: int
    settings: PclMinDetectableRcsCalcSettings
    values: list[list[list[float]]]


class SimulationResults(pydantic.BaseModel):
    monostaticCoverages: list[tuple[MonostaticCoverageCalculation, datetime.datetime]]
    """calculation result, date of calculation [epochs]"""
    pclMinDetectableRcsGrids: list[
        tuple[PclMinDetectableRcsCalculation, datetime.datetime]
    ]
    """calculation result, date of calculation [epochs]"""

    def to_annotations(self, sensor_id: int, target_id: int) -> list[GeoJSONAnnotation]:
        """
        Monostatic coverages of a sensor as annotations.

        `target_id` is the ID of the radar the sensor belongs to.
        """
        result: list[GeoJSONAnnotation] = []
        for calc, _ in self.monostaticCoverages:
            if calc.sensorId != sensor_id:
                continue
            alt = calc.settings.targetAlt
            rcs = calc.settings.targetRcs
            result.append(
                GeoJSONAnnotation(
                    id=f"sensor-range-{calc.sensorId}-{alt:g}-{rcs:g}",
                    feature=GeoJSONFeature(
                        geometry=GeoJSONMultiPolygon.merge(calc.coverage)
                    ),
                    kind=GeoJSONKind.SENSOR_RANGE,
                    target_ids=[target_id],
                    hover_text=f"Monostatic #{calc.sensorId}, {alt:.0f} MASL, RCS={rcs:.1f}m^2",
                )
            )
        return result

    def __add__(self, other: SimulationResults) -> SimulationResults:
        return SimulationResults.model_construct(
            monostaticCoverages=self.monostaticCoverages + other.monostaticCoverages,
            pclMinDetectableRcsGrids=self.pclMinDetectableRcsGrids
            + other.pclMinDetectableRcsGrids,
        )


class OrderOfBattle(pydantic.BaseModel):
    monostatic_sensors: list[MonostaticSensorFactory]
    pcl_sensors: list[PclSensorFactory]
    gbads: list[StaticGbadFactory]
    simulationResults: SimulationResults
    oneway_drones: list[FixedPathOneWayDroneFactory]
    ballistic_missiles: list[BallisticMissileFactory]
    critical_infrastructure: list[CriticalInfrastructureFactory] = []
    cruise_missiles: list[CruiseMissileFactory] = []
    camp_sites: list[CampSite] = []
    """Sites with counted stock."""
    transport_times: list[TransportTime] = []
    """Routes along which stock can be moved between camp sites"""
    intel: list[IntelFactory] = []
    """Intel sources of this side"""
    unused_id_sensor: int
    unused_id_receiver: int
    unused_id_transmitter: int
    unused_id_effector: int
    unused_target_id: int
    unused_id_camp_site: int = 0
    """
    Next free camp site ID. Defaults to 0 for ORBAT files written before camp
    sites existed, which have none.
    """
    is_user_controlled: bool = False
    """
    Whether a player commands this side with actions. If so, ``to_controller``
    returns a ``ActionController``. Defaults to False for ORBAT files written
    before players existed.
    """

    @pydantic.model_validator(mode="after")
    def _check_camp_sites(self) -> OrderOfBattle:
        sites: dict[int, CampSite] = {}
        for site in self.camp_sites:
            if site.id in sites:
                raise ValueError(f"Duplicate camp site ID: {site.id}")
            sites[site.id] = site
        for route in self.transport_times:
            for site_id in (route.source_site_id, route.target_site_id):
                if site_id not in sites:
                    raise ValueError(
                        f"Transport route refers to unknown camp site {site_id}"
                    )
            source = sites[route.source_site_id]
            target = sites[route.target_site_id]
            if source.ammo.type != target.ammo.type:
                raise ValueError(
                    f"Transport route {source.id} -> {target.id} connects "
                    f"different ammunition types ({source.ammo.type} and "
                    f"{target.ammo.type})"
                )
        if self.camp_sites and not self.is_user_controlled:
            logging.warning(
                f"Order of battle has {len(self.camp_sites)} camp site(s) but is "
                "not user-controlled: the camp sites are simulated as targets "
                "only; nobody can launch from them or move their stock."
            )
        return self

    @staticmethod
    def from_file(orbat_file: str) -> OrderOfBattle:
        with open(orbat_file, "r") as file:
            data = json.load(file)
            orbat = OrderOfBattle.model_validate(data)
        # Validate IDs.
        id_provider = IdProvider()
        orbat.update_id_provider(id_provider)
        consistent_ids = (
            orbat.unused_id_sensor >= id_provider._free_ids[Entity.SENSOR]
            and orbat.unused_id_receiver >= id_provider._free_ids[Entity.RECEIVER]
            and orbat.unused_id_transmitter >= id_provider._free_ids[Entity.TRANSMITTER]
            and orbat.unused_id_effector >= id_provider._free_ids[Entity.EFFECTOR]
            and orbat.unused_target_id >= id_provider._free_ids[Entity.TARGET]
            and orbat.unused_id_camp_site >= id_provider._free_ids[Entity.CAMP_SITE]
        )
        if not consistent_ids:
            raise ValueError("ORBAT file is inconsistent: Unused IDs are used")
        return orbat

    def to_controller(
        self,
        terrain: AbstractTerrainModel,
        is_blue: bool,
    ) -> ControllerGroup:
        monostatic_controllers = [
            s.to_controller(
                is_blue,
                self.simulationResults.to_annotations(s.sensor.id, s.target_id),
            )
            for s in self.monostatic_sensors
        ]
        if self.is_user_controlled:
            # The player can switch the radars on and off.
            monostatic_controllers = [
                LivingController(
                    child=SwitchableMonostaticRadarController(
                        child=living.child,
                        sensor_id=factory.sensor.id,
                    ),
                    target_id=living.target_id,
                )
                for factory, living in zip(
                    self.monostatic_sensors, monostatic_controllers, strict=True
                )
            ]

        ##########################
        # TODO: Load PCL sensors.
        ##########################

        static_deployment_controllers = [
            StaticGbadCoordinator(
                controllers=[
                    LivingController(
                        child=e.to_controller(terrain, is_blue),
                        target_id=e.target_id,
                    )
                    for e in self.gbads
                ],
                terrain=terrain,
            )
        ]

        oneway_drone_controllers = [
            drone.to_controller(terrain, is_blue) for drone in self.oneway_drones
        ]
        bm_controllers = [bm.to_controller(is_blue) for bm in self.ballistic_missiles]
        cm_controllers = [cm.to_controller(is_blue) for cm in self.cruise_missiles]
        infra_controllers = [
            c.to_controller(is_blue) for c in self.critical_infrastructure
        ]
        camp_site_controllers = [s.to_controller(is_blue) for s in self.camp_sites]
        intel_controllers = [i.to_controller(is_blue) for i in self.intel]

        tree = GeoJsonController(
            child=ControllerGroup(
                controllers=monostatic_controllers
                + static_deployment_controllers
                + oneway_drone_controllers
                + bm_controllers
                + cm_controllers
                + infra_controllers
                + camp_site_controllers
                + intel_controllers
            ),
            geojson_features=[],
        )
        if not self.is_user_controlled:
            return ControllerGroup(controllers=[tree])
        return ActionController(
            controllers=[tree],
            is_blue=is_blue,
            terrain=terrain,
            camp_sites=camp_site_controllers,
            transport_times=self.transport_times,
            radars=monostatic_controllers,
        )

    def update_id_provider(self, id_provider: IdProvider):
        for detectable_sensor in self.monostatic_sensors:
            sensor = detectable_sensor.sensor
            id_provider.register_entity(Entity.SENSOR, sensor.id)
            id_provider.register_entity(Entity.TRANSMITTER, sensor.transmitter.id)
            id_provider.register_entity(Entity.RECEIVER, sensor.receiver.id)
            id_provider.register_entity(Entity.TARGET, detectable_sensor.target_id)
        for detectable_sensor in self.pcl_sensors:
            sensor = detectable_sensor.sensor
            if sensor.receiver.id in id_provider._used_ids[Entity.RECEIVER]:
                raise ValueError(f"PCL Receiver ID duplicated: {sensor.receiver.id}")
            if sensor.transmitter.id in id_provider._used_ids[Entity.TRANSMITTER]:
                raise ValueError(
                    f"PCL Transmitter ID duplicated: {sensor.transmitter.id}"
                )
        for detectable_sensor in self.pcl_sensors:
            sensor = detectable_sensor.sensor
            id_provider.register_entity(Entity.SENSOR, sensor.id)
            # Receiver and Transmitter can be duplicated in PCL sensors.
            try:
                id_provider.register_entity(Entity.TRANSMITTER, sensor.transmitter.id)
            except ValueError:
                pass
            try:
                id_provider.register_entity(Entity.RECEIVER, sensor.receiver.id)
            except ValueError:
                pass
        for gbad_factory in self.gbads:
            effector = gbad_factory.gbad
            id_provider.register_entity(Entity.EFFECTOR, effector.id)
            id_provider.register_entity(Entity.TARGET, gbad_factory.target_id)
            # effector.projectile.id (if indirect) is intentionally not
            # registered here. The projectile is only a template. Each shot
            # deep-copies it and mints a fresh id. The template's id is never
            # actually used for identity.
        for drone in self.oneway_drones:
            id_provider.register_entity(Entity.EFFECTOR, drone.effector.id)
            id_provider.register_entity(Entity.TARGET, drone.trajectory.target_id)
        for missile in self.ballistic_missiles:
            id_provider.register_entity(Entity.EFFECTOR, missile.effector_id)
            id_provider.register_entity(Entity.TARGET, missile.target_id)
        for infra in self.critical_infrastructure:
            id_provider.register_entity(Entity.TARGET, infra.target_id)
        for missile in self.cruise_missiles:
            id_provider.register_entity(Entity.EFFECTOR, missile.effector_id)
            id_provider.register_entity(Entity.TARGET, missile.target_id)
        for site in self.camp_sites:
            id_provider.register_entity(Entity.CAMP_SITE, site.id)
            id_provider.register_entity(Entity.TARGET, site.target_id)

    def reindex(self, id_provider: IdProvider):
        sensor_id_mapping: dict[int, int] = {}
        for detectable_sensor in self.monostatic_sensors:
            detectable_sensor.target_id = id_provider.increment(Entity.TARGET)
            s = detectable_sensor.sensor
            sensor_id_mapping[s.id] = id_provider.increment(Entity.SENSOR)
            s.id = sensor_id_mapping[s.id]
            s.transmitter.id = id_provider.increment(Entity.TRANSMITTER)
            s.receiver.id = id_provider.increment(Entity.RECEIVER)
        pcl_receiver_id_mapping: dict[int, int] = {}
        for detectable_sensor in self.pcl_sensors:
            detectable_sensor.target_id = id_provider.increment(Entity.TARGET)
            s = detectable_sensor.sensor
            sensor_id_mapping[s.id] = id_provider.increment(Entity.SENSOR)
            s.id = sensor_id_mapping[s.id]
            s.transmitter.id = id_provider.increment(Entity.TRANSMITTER)

            # Make sure that the receiver ID is only updated once and kept consistent.
            if s.receiver.id in pcl_receiver_id_mapping:
                s.receiver.id = pcl_receiver_id_mapping[s.receiver.id]
            else:
                new_id = id_provider.increment(Entity.RECEIVER)
                pcl_receiver_id_mapping[s.receiver.id] = new_id
                s.receiver.id = new_id
        for gbad_factory in self.gbads:
            gbad_factory.target_id = id_provider.increment(Entity.TARGET)
            gbad_factory.gbad.id = id_provider.increment(Entity.EFFECTOR)
            # The projectile template's id is deliberately left untouched
            # because it is only a placeholder that is never used.
        for calc, t in self.simulationResults.monostaticCoverages:
            calc.sensorId = sensor_id_mapping[calc.sensorId]
        for calc, t in self.simulationResults.pclMinDetectableRcsGrids:
            calc.sensor_id = sensor_id_mapping[calc.sensor_id]
        for drone in self.oneway_drones:
            drone.effector.id = id_provider.increment(Entity.EFFECTOR)
            new_id = id_provider.increment(Entity.TARGET)
            print(
                f"Update trajectory targe_id {drone.trajectory.target_id} -> {new_id}"
            )
            drone.trajectory.target_id = new_id
        for missile in self.ballistic_missiles:
            missile.effector_id = id_provider.increment(Entity.EFFECTOR)
            missile.target_id = id_provider.increment(Entity.TARGET)
        for infra in self.critical_infrastructure:
            infra.target_id = id_provider.increment(Entity.TARGET)
        for missile in self.cruise_missiles:
            missile.effector_id = id_provider.increment(Entity.EFFECTOR)
            missile.target_id = id_provider.increment(Entity.TARGET)
        site_id_mapping: dict[int, int] = {}
        for site in self.camp_sites:
            site_id_mapping[site.id] = id_provider.increment(Entity.CAMP_SITE)
            site.id = site_id_mapping[site.id]
            site.target_id = id_provider.increment(Entity.TARGET)
        for route in self.transport_times:
            route.source_site_id = site_id_mapping[route.source_site_id]
            route.target_site_id = site_id_mapping[route.target_site_id]

    def _trajectory_time_spans(
        self,
    ) -> list[tuple[datetime.datetime, datetime.datetime]]:
        trajectories = (
            [drone.trajectory for drone in self.oneway_drones]
            + [bm.get_trajectory(True) for bm in self.ballistic_missiles]
            + [cm.get_trajectory(True) for cm in self.cruise_missiles]
        )
        return [(t.times[0], t.times[-1]) for t in trajectories]

    @property
    def t_min(self) -> datetime.datetime | None:
        spans = self._trajectory_time_spans()
        if len(spans) == 0:
            return None
        return min(start for start, _ in spans)

    @property
    def t_max(self) -> datetime.datetime | None:
        spans = self._trajectory_time_spans()
        if len(spans) == 0:
            return None
        return max(stop for _, stop in spans)

    @staticmethod
    def merge(
        orbat1: OrderOfBattle,
        orbat2: OrderOfBattle,
    ) -> OrderOfBattle:
        id_provider = IdProvider()
        orbat1.update_id_provider(id_provider)
        orbat2.reindex(id_provider)

        return OrderOfBattle(
            monostatic_sensors=orbat1.monostatic_sensors + orbat2.monostatic_sensors,
            pcl_sensors=orbat1.pcl_sensors + orbat2.pcl_sensors,
            gbads=orbat1.gbads + orbat2.gbads,
            simulationResults=orbat1.simulationResults + orbat2.simulationResults,
            oneway_drones=orbat1.oneway_drones + orbat2.oneway_drones,
            ballistic_missiles=orbat1.ballistic_missiles + orbat2.ballistic_missiles,
            critical_infrastructure=orbat1.critical_infrastructure
            + orbat2.critical_infrastructure,
            cruise_missiles=orbat1.cruise_missiles + orbat2.cruise_missiles,
            camp_sites=orbat1.camp_sites + orbat2.camp_sites,
            transport_times=orbat1.transport_times + orbat2.transport_times,
            intel=orbat1.intel + orbat2.intel,
            unused_id_sensor=id_provider.increment(Entity.SENSOR),
            unused_id_receiver=id_provider.increment(Entity.RECEIVER),
            unused_id_transmitter=id_provider.increment(Entity.TRANSMITTER),
            unused_id_effector=id_provider.increment(Entity.EFFECTOR),
            unused_target_id=id_provider.increment(Entity.TARGET),
            unused_id_camp_site=id_provider.increment(Entity.CAMP_SITE),
            is_user_controlled=orbat1.is_user_controlled or orbat2.is_user_controlled,
        )

    @staticmethod
    def to_openapi_schema() -> dict:
        schema = OrderOfBattle.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        defs = schema.pop("$defs", {})
        return {
            "openapi": "3.1.0",
            "info": {"title": "Theia", "version": theia.__version__},
            "paths": {},
            "components": {
                "schemas": {
                    "OrderOfBattle": schema,
                    **defs,
                },
            },
        }


class PseudoTrackerParams(pydantic.BaseModel):
    removal_patience: int
    start_timestamp: float
    prior_position: Point


class TrackerFactory(pydantic.BaseModel):
    tracker_name: Literal["pseudotracker"] = "pseudotracker"
    parameters: PseudoTrackerParams

    def to_tracker(self, rng: np.random.Generator) -> AbstractTracker:
        if self.tracker_name == "pseudotracker":
            return PseudoTracker(
                removal_patience=self.parameters.removal_patience,
                rng=rng,
                start_time=datetime.datetime.fromtimestamp(
                    self.parameters.start_timestamp,
                    tz=datetime.UTC,
                ),
                prior_position=self.parameters.prior_position,
            )
        else:
            raise ValueError(f"Unknown tracker '{self.tracker_name}'")


class DamageModelFactory(pydantic.BaseModel):
    model_name: Literal["kill_always"]

    def to_damage_model(self, rng: np.random.Generator) -> AbstractDamageModel:
        if self.model_name == "kill_always":
            return UniformDamageModel(p_kill=1.0, rng=rng)
        else:
            raise RuntimeError(f"Unknown damage model {self.model_name}")


class TimeTermination(pydantic.BaseModel):
    """Stop the simulation after ``stop_time``."""

    type: Literal["time"] = "time"
    stop_time: datetime.datetime

    def to_criterion(self) -> TerminationCriterion:
        return TimeCriterion(end_time=self.stop_time)


class NeverTermination(pydantic.BaseModel):
    """Run until the simulation is stopped from outside, e. g. for a game."""

    type: Literal["never"] = "never"

    def to_criterion(self) -> TerminationCriterion:
        return NeverCriterion()


Termination = Annotated[
    TimeTermination | NeverTermination,
    pydantic.Field(discriminator="type"),
]


class ScenarioFactory(pydantic.BaseModel):
    name: str
    start_time: datetime.datetime
    """Start time of the scenario"""
    stop_time: datetime.datetime
    """Start time of the scenario"""
    time_step: int
    """Time step per iteration [s]"""
    blue_orbat: OrderOfBattle
    red_orbat: OrderOfBattle
    blue_tracker: TrackerFactory
    red_tracker: TrackerFactory
    terrain_model: TerrainFactory
    damage_model: DamageModelFactory
    seed: int
    blue_geojson: dict[str, list[GeoJSONFeature]] = {}
    """List of GeoJSON objects for BLUE that belong to the same tag."""
    red_geojson: dict[str, list[GeoJSONFeature]] = {}
    """List of GeoJSON objects for RED that belong to the same tag."""
    termination: Termination | None = None
    """When the simulation stops. ``None`` means after ``stop_time``."""

    def termination_criterion(self) -> TerminationCriterion:
        if self.termination is None:
            return TimeCriterion(end_time=self.stop_time)
        return self.termination.to_criterion()

    def update_id_provider(self, id_provider: IdProvider):
        self.blue_orbat.update_id_provider(id_provider)
        self.red_orbat.update_id_provider(id_provider)

    def to_simulator(
        self,
        output_path: str,
        is_interactive: bool,
        debugging: bool,
    ) -> tuple[Simulator, SituationalPictureBuffer | None]:
        id_provider = IdProvider()

        # Ensure consistent IDs.
        self.update_id_provider(id_provider)

        terrain = self.terrain_model.to_terrain()

        controller_blue = self.blue_orbat.to_controller(terrain, True)
        controller_red = self.red_orbat.to_controller(terrain, False)

        buffer = None
        listener = FileLogger(
            path=output_path,
            log_situational_picture=debugging,
        )

        if is_interactive:
            buffer = SituationalPictureBuffer()
            listener = CompositeSimulationListener([listener, buffer])

        pcl_detector = PclDetector()
        pet_detector = PetDetector(terrain_model=terrain)

        rng = np.random.Generator(np.random.PCG64(seed=self.seed))

        min_time_step = datetime.timedelta(seconds=1 if is_interactive else 0)

        simulator = Simulator(
            pcl_detector=pcl_detector,
            pet_detector=pet_detector,
            visual_detector=None,
            blue_controller=controller_blue,
            red_controller=controller_red,
            blue_tracker=self.blue_tracker.to_tracker(rng),
            red_tracker=self.red_tracker.to_tracker(rng),
            start_time=self.start_time,
            time_step=datetime.timedelta(seconds=self.time_step),
            termination_criterion=self.termination_criterion(),
            min_time_per_step=min_time_step,
            rng=rng,
            listener=listener,
            terrain_model=terrain,
            damage_model=self.damage_model.to_damage_model(rng),
            simulate_clutter=False,
            id_provider=id_provider,
            shot_association_tolerance=250.0,
        )

        return simulator, buffer


class BallisticMissileFactory(pydantic.BaseModel):
    p_start: Point
    p_stop: Point
    t_start: datetime.datetime
    terrain: TerrainFactory
    target_id: int
    effector_id: int
    rcs: float
    alpha: float | None = 45.0
    """Launch angle w. r. t. LOS [°]"""
    category: TargetCategory = TargetCategory.SHORT_RANGE_BALLISTIC_MISSILE

    def model_post_init(self, __context):
        self._v0, self._times, self._trajectory_data = build_trajectory(
            np.deg2rad(self.alpha),
            self.p_start,
            self.p_stop,
        )

    def to_controller(self, is_blue: bool) -> LivingController[FixedPathOneWayDrone]:
        bm = FixedPathOneWayDrone(
            effector=DirectFireEffector(
                id=self.effector_id,
                name="ballistic missile",
                point=self.p_start,
                combat_range=BM_COMBAT_RANGE,
                n_attacks_left=1,
                terrain=self.terrain.to_terrain(),
                # One-shot missile - no rate-of-fire limit applies.
                cadence=float("inf"),
            ),
            trajectory=self.get_trajectory(is_blue),
            assigned_goal=self.p_stop,
            terrain=self.terrain.to_terrain(),
        )
        return LivingController(child=bm, target_id=bm.trajectory.target_id)

    def get_trajectory(self, is_blue: bool) -> Trajectory:
        points = convert_to_geodetic(self.p_start, self.p_stop, self._trajectory_data)
        return Trajectory(
            target_id=self.target_id,
            target_info=TargetInfos.for_category(self.category).with_party(
                Party.from_is_blue(is_blue)
            ),
            times=[
                self.t_start + datetime.timedelta(seconds=float(t)) for t in self._times
            ],
            lats=[p.lat for p in points],
            lons=[p.lon for p in points],
            alts=[p.alt for p in points],
            cross_section_model=ConstantRcsModel(rcs=self.rcs),
        )

    def plot_trajectory(self) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        fig, ax = matplotlib.pyplot.subplots()
        data = _build_earth_curvature(self.p_start, self.p_stop) / 1e3
        ax.plot(
            data[:, 0],
            data[:, 1],
            "-",
            label="earth surface at sea level",
            color="black",
        )
        data = _build_earth_curvature(self.p_start, self.p_stop, 100_000) / 1e3
        ax.plot(
            data[:, 0],
            data[:, 1],
            "--",
            label="Karman line ('space')",
            color="black",
        )

        ax.plot(
            self._trajectory_data[:, 0] / 1e3,
            self._trajectory_data[:, 1] / 1e3,
            "-",
            label=rf"Trajectory for $\alpha = {self.alpha} \degree$; T = {self._times[-1] / 60:.1f}min; $v_0$ = {self._v0:.0f} m/s",
        )

        ax.set_xlabel("LOS distance [km]", fontsize=14)
        ax.set_ylabel("Altitude above LOS [km]", fontsize=14)
        ax.legend()
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        return fig, ax


class CruiseMissileFactory(pydantic.BaseModel):
    """
    Terrain-following cruise missile, see :mod:`theia.cruise_missile`.

    The altitudes of ``p_start`` and ``p_stop`` are ignored: the missile starts
    at least ``cruise_magl`` above the terrain (higher if needed to clear the
    terrain right after the launch) and hits the terrain at the target.
    """

    p_start: Point
    p_stop: Point
    t_start: datetime.datetime
    terrain: TerrainFactory
    target_id: int
    effector_id: int
    rcs: float
    speed: float
    """Constant speed along the path [m/s]"""
    cruise_magl: float
    """Nominal cruise height above ground level [m]"""
    min_flight_path_angle: float = CM_DEFAULT_MIN_FLIGHT_PATH_ANGLE
    """Steepest descent while cruising [°], in (-90°, 0°)"""
    max_flight_path_angle: float = CM_DEFAULT_MAX_FLIGHT_PATH_ANGLE
    """Steepest climb while cruising [°], in (0°, 90°)"""
    terminal_dive_angle: float = CM_DEFAULT_TERMINAL_DIVE_ANGLE
    """Flight-path angle of the terminal dive [°], in (-90°, 0°) (negative = descending)"""
    min_clearance: Optional[float] = None
    """Minimum clearance above the terrain while cruising [m].
    ``None`` means ``min(cruise_magl, max(15 m, 0.5 * cruise_magl))``."""
    sample_spacing: float = CM_DEFAULT_SAMPLE_SPACING
    """Distance between trajectory samples along the ground track [m]"""
    category: TargetCategory = TargetCategory.CRUISE_MISSILE

    _path: CruiseMissilePath | None = pydantic.PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def _check_parameters(self) -> CruiseMissileFactory:
        validate_parameters(
            self.p_start,
            self.p_stop,
            self.speed,
            self.cruise_magl,
            self.min_flight_path_angle,
            self.max_flight_path_angle,
            self.terminal_dive_angle,
            self.effective_min_clearance,
            self.sample_spacing,
        )
        # Validate trajectory.
        self.path
        return self

    @property
    def effective_min_clearance(self) -> float:
        if self.min_clearance is None:
            return default_min_clearance(self.cruise_magl)
        return self.min_clearance

    @property
    def path(self) -> CruiseMissilePath:
        # Computed lazily because it requires terrain lookups.
        if self._path is None:
            self._path = build_terrain_following_path(
                p_start=self.p_start,
                p_stop=self.p_stop,
                speed=self.speed,
                cruise_magl=self.cruise_magl,
                terrain=self.terrain.to_terrain(),
                min_flight_path_angle=self.min_flight_path_angle,
                max_flight_path_angle=self.max_flight_path_angle,
                terminal_dive_angle=self.terminal_dive_angle,
                min_clearance=self.effective_min_clearance,
                sample_spacing=self.sample_spacing,
            )
        return self._path

    def to_controller(self, is_blue: bool) -> LivingController[FixedPathOneWayDrone]:
        missile = FixedPathOneWayDrone(
            effector=DirectFireEffector(
                id=self.effector_id,
                name="cruise missile",
                point=self.p_start,
                combat_range=CM_COMBAT_RANGE,
                n_attacks_left=1,
                terrain=self.terrain.to_terrain(),
                # One-shot missile - no rate-of-fire limit applies.
                cadence=float("inf"),
            ),
            trajectory=self.get_trajectory(is_blue),
            assigned_goal=self.path.impact,
            terrain=self.terrain.to_terrain(),
        )
        return LivingController(child=missile, target_id=self.target_id)

    def get_trajectory(self, is_blue: bool) -> Trajectory:
        path = self.path
        return Trajectory(
            target_id=self.target_id,
            target_info=TargetInfos.for_category(self.category).with_party(
                Party.from_is_blue(is_blue)
            ),
            times=[
                self.t_start + datetime.timedelta(seconds=float(t)) for t in path.times
            ],
            lats=path.lats.tolist(),
            lons=path.lons.tolist(),
            alts=path.alts.tolist(),
            cross_section_model=ConstantRcsModel(rcs=self.rcs),
        )


def _build_earth_curvature(p_start, p_stop, altitude=0):
    distance = line_of_sight_distance(*p_start.as_tuple(), *p_stop.as_tuple())

    p_start_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_start.as_tuple())
    )
    p_stop_ecef = np.array(
        CoordinateTransformations.geodetic_to_cartesian(*p_stop.as_tuple())
    )
    direction = p_stop_ecef - p_start_ecef
    direction /= np.linalg.norm(direction)

    xs = list(np.arange(0, distance, 100))
    ys = []
    for d in xs:
        p_los_ecef = p_start_ecef + d * direction
        p_los_geodetic = CoordinateTransformations.cartesian_to_geodetic(*p_los_ecef)
        y = altitude - p_los_geodetic[2]
        ys.append(y)

    return np.vstack([xs, ys]).T


def build_scenario(
    scenario_name: str,
    terrain_factory: TerrainFactory,
    seed: int,
    orbat_blue: OrderOfBattle,
    orbat_red: OrderOfBattle,
) -> ScenarioFactory:
    id_provider = IdProvider()
    orbat_blue.update_id_provider(id_provider)
    orbat_red.reindex(id_provider)

    t_min = None
    t_max = None
    if orbat_blue.t_min is not None:
        t_min = orbat_blue.t_min
        t_max = orbat_blue.t_max
    if orbat_red.t_min is not None:
        t_min = orbat_red.t_min if t_min is None else min(t_min, orbat_red.t_min)
        t_max = orbat_red.t_max if t_max is None else max(t_max, orbat_red.t_max)

    is_game = orbat_blue.is_user_controlled or orbat_red.is_user_controlled
    if t_min is None or t_max is None:
        if not is_game:
            raise ValueError("No mobile parts in the simulation!")
        # A game needs no scripted mobile parts: the players launch them.
        t_min = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
        t_max = t_min

    t_max = datetime.datetime.fromtimestamp(
        np.ceil(t_max.timestamp() + 1),
        datetime.UTC,
    )
    # A game runs until it is stopped from outside.
    termination = NeverTermination() if is_game else None

    tracker_params = PseudoTrackerParams(
        removal_patience=10,
        start_timestamp=t_min.timestamp(),
        prior_position=Point(
            lat=POSITIONS_OF_INTEREST["CH_CENTER"]["lat"],
            lon=POSITIONS_OF_INTEREST["CH_CENTER"]["lon"],
            alt=1000,
        ),
    )

    scenario = ScenarioFactory(
        name=scenario_name,
        start_time=t_min,
        stop_time=t_max,
        time_step=1,
        blue_orbat=orbat_blue,
        red_orbat=orbat_red,
        blue_tracker=TrackerFactory(
            tracker_name="pseudotracker",
            parameters=tracker_params,
        ),
        red_tracker=TrackerFactory(
            tracker_name="pseudotracker",
            parameters=tracker_params,
        ),
        terrain_model=terrain_factory,
        damage_model=DamageModelFactory(model_name="kill_always"),
        seed=seed,
        termination=termination,
    )
    return scenario
