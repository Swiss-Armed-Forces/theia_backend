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
from theia.terrain import elevationAt
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
    MonostaticSensor,
    PclMeasurementModel,
    PclSensor,
    Point,
)


class BodenseeMonostaticPclFactory(AbstractSimulatorFactory):
    def __init__(self, rng: np.random.Generator):
        self._rng = rng
        trajectory = build_single_target_from_Bodensee(alt=1000)
        self._trajectories = [trajectory]

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

    def _build_pcl_sensors(self, start_sensor_id: int) -> list[PclSensor]:
        pcl_rx_lat = 46.99166835
        pcl_rx_lon = 8.36833333
        point = Point(
            lat=pcl_rx_lat,
            lon=pcl_rx_lon,
            alt=elevationAt(pcl_rx_lat, pcl_rx_lon),
        )
        pcl_rx = build_pcl_receiver(rx_id=1, point=point)
        txs = load_bakom_ukw_transmitters(start_id=1)
        tx_ids = [943, 1151, 921, 1113]
        txs = [tx for tx in txs if tx.id in tx_ids]
        pcl_sensors: list[PclSensor] = []
        sensor_id = start_sensor_id
        for tx in txs:
            pcl_sensors.append(
                PclSensor(
                    id=sensor_id,
                    transmitter=tx,
                    receiver=pcl_rx,
                    error_model=PclMeasurementModel(
                        min_bistatic_range_uncertainty=200.0,
                        max_bistatic_range_uncertainty=200.0,
                        min_doppler_uncertainty=5.0,
                        max_doppler_uncertainty=5.0,
                    ),
                )
            )
            sensor_id += 1

    def _get_blue_controller(self) -> Controller:
        monostatic_positions = [
            (
                POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
                POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
            ),
            # (46.97947680813955, 8.254819833983184),  # Pilatus
            # (46.572082512448134, 8.829917042912648),  # Scopi
            # (46.101596555481564, 7.716021211924495),  # Weisshorn
            # (46.835925325283476, 9.794288419787135),  # Weissfluh
        ]
        radar_lat = POSITIONS_OF_INTEREST["Uetliberg"]["lat"]
        radar_lon = POSITIONS_OF_INTEREST["Uetliberg"]["lon"]
        point = Point(
            lat=radar_lat,
            lon=radar_lon,
            alt=elevationAt(radar_lat, radar_lon),
        )
        sensor_id = 0
        monostatic_radars: list[MonostaticSensor] = []
        for pos in monostatic_positions:
            point = Point(lat=pos[0], lon=pos[1], alt=elevationAt(pos[0], pos[1]))
            monostatic_radar = build_flores_monostatic_radar(point, sensor_id, 0, 0)
            monostatic_radars.append(monostatic_radar)
            sensor_id += 1
            monostatic_controllers = [
                MonostaticRadarController(
                    1 + i,
                    r,
                    True,
                    ConstantRcsModel(rcs=1.0),
                )
                for i, r in enumerate(monostatic_radars)
            ]
            pcl_sensors = self._build_pcl_sensors(sensor_id)
            pcl_controllers = [
                PclSensorController(
                    monostatic_controllers[-1]._target_id + 2 * i + 1,
                    monostatic_controllers[-1]._target_id + 2 * i + 2,
                    sensor,
                    True,
                    ConstantRcsModel(rcs=1.0),
                )
                for i, sensor in enumerate(pcl_sensors)
            ]
            return ControllerGroup(monostatic_controllers + pcl_controllers)

    def _get_red_controller(self) -> Controller:
        return ControllerGroup(
            [
                WaypointTargetController.from_trajectory(
                    t,
                    SIDC_RED_FIXED_WING,
                    sensor=build_fighter_jet_radar(1, 1, 1),
                )
                for t in self._trajectories
            ]
        )

    def _get_start_time(self) -> datetime.datetime:
        return min([t.times[0] for t in self._trajectories])

    def _get_timestep(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=1)

    def _get_termination_criterion(self) -> TerminationCriterion:
        stop_time = max([t.times[-1] for t in self._trajectories])
        return TimeCriterion(stop_time)
