import datetime

import numpy as np

from theia.config import SIDC_RED_FIXED_WING
from theia.coordinates import POSITIONS_OF_INTEREST
from theia.data_loading import load_bakom_ukw_transmitters
from theia.detection.pcl import PclDetector
from theia.detection.pet import PetDetector
from theia.maneuvers import ConstantSpeedStraightManeuver
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
        self._trajectory1 = build_single_target_from_Bodensee(target_id=0)
        self._trajectory2 = ConstantSpeedStraightManeuver(
            stop_point=Point(
                lat=47.2240,
                lon=9.7335,
                alt=1000,
            ),
            speed=300.0,
        ).to_trajectory(
            start_time=self._trajectory1.times[0],
            start_pos=Point(
                lat=47.6100,
                lon=7.4212,
                alt=1000,
            ),
            target_id=1,
            sidc=SIDC_RED_FIXED_WING,
            rcs=1.0,
        )

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

    def _get_pcl_sensors(self) -> list[PclSensor]:
        pcl_rx = build_pcl_receiver(
            2,
            Point(
                lat=47.0658,
                lon=8.5250,
                alt=SrtmTerrainModel().elevationAt(47.0658, 8.5250),
            ),
        )
        pcl_rx2 = build_pcl_receiver(
            3,
            Point(
                lat=47.5674,
                lon=8.1393,
                alt=SrtmTerrainModel().elevationAt(47.5674, 8.1393),
            ),
        )

        txs = load_bakom_ukw_transmitters()
        sensors: list[PclSensor] = []

        tx_ids1 = (920, 1150, 942)
        txs1 = [tx for tx in txs if tx.id in tx_ids1]
        for i, tx in enumerate(txs1):
            sensor = PclSensor(
                id=2 + i,
                transmitter=tx,
                receiver=pcl_rx,
                error_model=PclMeasurementModel(),
            )
            sensors.append(sensor)

        tx_ids2 = (421, 1222, 756)
        txs2 = [tx for tx in txs if tx.id in tx_ids2]
        for i, tx in enumerate(txs2):
            sensor = PclSensor(
                id=2 + i + len(tx_ids1),
                transmitter=tx,
                receiver=pcl_rx2,
                error_model=PclMeasurementModel(),
            )
            sensors.append(sensor)
        return sensors

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
                target_id=2,
                radar=blue_sensor,
                is_blue=True,
                rcs_model=ConstantRcsModel(rcs=1.0),
                name="Üetliberg radar",
            )
        ]

        sensors = self._get_pcl_sensors()
        rx_ids = {
            2: 3,
            3: 4,
        }
        tx_ids = {
            920: 5,
            1150: 6,
            942: 7,
            421: 8,
            1222: 9,
            756: 10,
        }
        used_receiver_ids = []
        for sensor in sensors:
            receiver_owned = sensor.receiver.id in used_receiver_ids
            c = PclSensorController(
                target_id_rx=rx_ids[sensor.receiver.id],
                target_id_tx=tx_ids[sensor.transmitter.id],
                sensor=sensor,
                is_blue=True,
                rcs_model=ConstantRcsModel(rcs=1.0),
                own_receiver=not receiver_owned,
                own_transmitter=True,
            )
            if not receiver_owned:
                used_receiver_ids.append(sensor.receiver.id)
            blue_controllers.append(c)

        return ControllerGroup(controllers=blue_controllers)

    def _get_red_controller(self) -> Controller:
        red_sensor = build_fighter_jet_radar(0, 0, 0)
        c1 = WaypointTargetController.from_trajectory(
            trajectory=self._trajectory1,
            sidc=SIDC_RED_FIXED_WING,
            name="Reconnaissance plane",
            sensor=red_sensor,
        )

        c2 = WaypointTargetController.from_trajectory(
            trajectory=self._trajectory2,
            sidc=SIDC_RED_FIXED_WING,
            name="Transit",
            sensor=None,
        )

        return ControllerGroup(controllers=[c1, c2])

    def _get_start_time(self) -> datetime.datetime:
        return min([t.times[0] for t in [self._trajectory1]])

    def _get_timestep(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=1)

    def _get_termination_criterion(self) -> TerminationCriterion:
        stop_time = max([t.times[-1] for t in [self._trajectory1]])
        return TimeCriterion(stop_time)
