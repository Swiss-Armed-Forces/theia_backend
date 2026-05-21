import datetime

import numpy as np

from theia.config import SIDC_RED_FIXED_WING
from theia.coordinates import POSITIONS_OF_INTEREST
from theia.data_loading import load_bakom_ukw_transmitters
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.simulation.controllers.controller_group import ControllerGroup
from theia.simulation.controllers.monostatic_radar_controller import (
    MonostaticRadarController,
)
from theia.simulation.controllers.pcl_sensor_controller import PclSensorController
from theia.simulation.controllers.waypoint_target_controller import (
    WaypointTargetController,
)
from theia.simulation.factories.abstract_simulator_factory import (
    AbstractSimulatorFactory,
)
from theia.simulation.pseudo_tracker import PseudoTracker
from theia.simulation.simulator import TerminationCriterion, TimeCriterion
from theia.terrain import SrtmTerrainModel
from theia.test_data import (
    build_fighter_jet_radar,
    build_flores_monostatic_radar,
    build_pcl_receiver,
    build_single_target_from_Bodensee,
)
from theia.types import (
    AbstractTracker,
    ConstantRcsModel,
    Controller,
    PclMeasurementModel,
    PclSensor,
    Point,
)


class PerformanceDemoFactory(AbstractSimulatorFactory):
    def __init__(self, rng: np.random.Generator):
        self._rng = rng
        self._trajectory = build_single_target_from_Bodensee(target_id=0)

    def _get_pcl_detector(self) -> PclDetector:
        return PclDetector()

    def _get_pet_detector(self) -> PetDetector:
        return PetDetector()

    def _get_blue_tracker(self) -> AbstractTracker:
        return PseudoTracker(
            removal_patience=30,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_red_tracker(self) -> AbstractTracker:
        return PseudoTracker(
            removal_patience=30,
            rng=self._rng,
            start_time=self._get_start_time(),
        )

    def _get_blue_controller(self) -> Controller:
        blue_sensor = build_flores_monostatic_radar(
            Point(
                lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
                lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
                alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
            ),
            1,
            1,
            1,
        )
        blue_controllers = [
            MonostaticRadarController(
                target_id=1,
                radar=blue_sensor,
                is_blue=True,
                rcs_model=ConstantRcsModel(rcs=1.0),
                name="Üetliberg radar",
            )
        ]

        pcl_rx = build_pcl_receiver(
            2,
            Point(
                lat=47.0658,
                lon=8.5250,
                alt=SrtmTerrainModel().elevationAt(47.0658, 8.5250),
            ),
        )

        txs = load_bakom_ukw_transmitters()
        txs = [tx for tx in txs if tx.id in (920, 1150, 942)]

        for i, tx in enumerate(txs):
            sensor = PclSensor(
                id=2 + i,
                transmitter=tx,
                receiver=pcl_rx,
                error_model=PclMeasurementModel(),
            )
            c = PclSensorController(
                target_id_rx=2,
                target_id_tx=3 + i,
                sensor=sensor,
                is_blue=True,
                rcs_model=ConstantRcsModel(rcs=1.0),
                own_receiver=i == 0,
                own_transmitter=True,
            )
            blue_controllers.append(c)

        return ControllerGroup(controllers=blue_controllers)

    def _get_red_controller(self) -> Controller:
        red_sensor = build_fighter_jet_radar(0, 0, 0)
        return WaypointTargetController.from_trajectory(
            trajectory=self._trajectory,
            sidc=SIDC_RED_FIXED_WING,
            name="Reconnaissance plane",
            sensor=red_sensor,
        )

    def _get_start_time(self) -> datetime.datetime:
        return min([t.times[0] for t in [self._trajectory]])

    def _get_timestep(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=1)

    def _get_termination_criterion(self) -> TerminationCriterion:
        stop_time = max([t.times[-1] for t in [self._trajectory]])
        return TimeCriterion(stop_time)
