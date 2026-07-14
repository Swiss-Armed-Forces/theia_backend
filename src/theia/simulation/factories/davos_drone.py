import datetime
import json

import numpy as np

from theia.config import SIDC
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.effectors import DirectFireEffector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.fixed_path_one_way_drone import (
    FixedPathOneWayDrone,
)
from theia.simulation.controllers.living_controller import LivingController
from theia.simulation.controllers.stationary_eye_controller import (
    StationaryEyeController,
)
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.factories.abstract_simulator_factory import (
    AbstractSimulatorFactory,
)
from theia.simulation.simulator import TerminationCriterion, TimeCriterion
from theia.simulation.trackers.pseudo_tracker import PseudoTracker
from theia.terrain import AbstractTerrainModel
from theia.types import (
    AbstractTracker,
    ConstantRcsModel,
    Controller,
    Entity,
    IdProvider,
    Point,
    Receiver,
    Trajectory,
    VisualSensor,
)


class DavosDroneScenarioFactory(AbstractSimulatorFactory):
    def __init__(self, rng: np.random.Generator, terrain_model: AbstractTerrainModel):
        self._rng = rng
        self._terrain_model = terrain_model
        self._t0 = datetime.datetime(
            year=2026,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
        )
        self._tmax = self._t0 + datetime.timedelta(hours=1)

        self.BLUE_INFRA_LAT = 46.80089
        self.BLUE_INFRA_LON = 9.83124
        self._p_infra = Point(
            lat=self.BLUE_INFRA_LAT,
            lon=self.BLUE_INFRA_LON,
            alt=self._terrain_model.elevationAt(
                self.BLUE_INFRA_LAT, self.BLUE_INFRA_LON
            )
            + 10,
        )

        # Load trajectory.
        data_dir = "/home/user/Documents/theia_backend/examples/conference/data"
        with open(f"{data_dir}/flight-path-spacetime_adversarial.json", "r") as file:
            data = json.load(file)
            self._times = [
                self._t0 + datetime.timedelta(seconds=p["time"] - 1550)
                for p in data
                if p["time"] > 1550
            ]
            self._points = [
                Point(lat=p["lat"], lon=p["lon"], alt=1600)
                for p in data
                if p["time"] > 1550
            ]
        self._id_provider = IdProvider()

    def _get_pcl_detector(self) -> PclDetector:
        return PclDetector()

    def _get_pet_detector(self) -> PetDetector:
        return PetDetector(terrain_model=self._terrain_model)

    def _get_blue_tracker(self) -> AbstractTracker:
        return PseudoTracker(
            removal_patience=30,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_red_tracker(self) -> AbstractTracker:
        return PseudoTracker(
            removal_patience=0,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_blue_controller(self) -> Controller:
        # Define critical infrastructure.
        target_id = self._id_provider.increment(Entity.TARGET)

        return LivingController(
            child=WaypointTargetController(
                name="Conference center",
                sidc=SIDC.BLUE_GOVERNMENT_SITE,
                target_id=self._id_provider.increment(Entity.TARGET),
                times=[self._t0, self._tmax],
                waypoints=[self._p_infra, self._p_infra],
                rcs_model=ConstantRcsModel(rcs=100),
            ),
            target_id=target_id,
        )

    def _get_red_controller(self) -> Controller:
        trajectory = Trajectory(
            target_id=self._id_provider.increment(Entity.TRACK),
            target_sidc=SIDC.RED_FIXED_WING,
            times=self._times,
            lats=[p.lat for p in self._points],
            lons=[p.lon for p in self._points],
            alts=[p.alt for p in self._points],
            vxs=[0.0 for _ in range(len(self._points))],
            vys=[0.0 for _ in range(len(self._points))],
            vzs=[0.0 for _ in range(len(self._points))],
            cross_section_model=ConstantRcsModel(rcs=1.0),
        )
        effector = DirectFireEffector(
            id=self._id_provider.increment(Entity.EFFECTOR),
            name="kamikaze drone",
            point=self._p_infra,
            combat_range=1000.0,
            n_attacks_left=1,
            terrain=self._terrain_model,
        )
        controller = LivingController(
            child=FixedPathOneWayDrone(
                effector=effector,
                trajectory=trajectory,
                assigned_goal=self._p_infra,
                terrain=self._terrain_model,
            ),
            target_id=trajectory.target_id,
        )
        # We know the position of the conference center.
        conf_center_observer = StationaryEyeController(
            sensor=VisualSensor(
                id=self._id_provider.increment(Entity.SENSOR),
                receiver=Receiver(
                    id=0,
                    point=self._p_infra,
                    antenna_height=0.0,
                    diameter=0.0,
                    cpi_pulses=0,
                    pfa=0.0,
                    min_elevation=0.0,
                    max_elevation=np.pi / 2.0,
                    min_azimuth=0.0,
                    max_azimuth=2 * np.pi,
                    rotation_time=0,
                    bandwidth=0,
                ),
                detection_range=10.0,
            ),
        )
        return ControllerGroup(controllers=[controller, conf_center_observer])

    def _get_start_time(self) -> datetime.datetime:
        return self._t0

    def _get_timestep(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=1)

    def _get_termination_criterion(self) -> TerminationCriterion:
        return TimeCriterion(self._tmax)
