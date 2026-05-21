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
        import numpy as np
        from theia.types import Polarization, Transmitter
        from theia.util import to_dB

        tx_locs = np.array(
            [
                [47.42035, 7.95848, 954.34130859],
                [47.47835, 7.64948, 756.82000732],
                [47.53035, 8.17048, 697.52001953],
                [47.68535, 8.29048, 704.26000977],
                [47.65835, 7.96348, 1038.01635742],
                [47.22435, 8.02548, 761.62097168],
            ]
        )

        txs = []
        for i, (lat, lon, alt) in enumerate(tx_locs):
            p = Point(
                lat=lat,
                lon=lon,
                alt=alt,
            )
            tx = Transmitter(
                id=2 + i,
                point=p,
                power=100_000,
                erp=to_dB(10_000),
                antenna_height=8,
                antenna_diameter=2,
                antenna_gain=0,
                frequency=95.0,
                pulse_width=0.0,
                bandwidth=0.190,
                polarization=Polarization.VERTICAL,
            )
            txs.append(tx)

        rx_locs = [
            [47.42535, 8.19448, 636.02301025],
            [47.56635, 7.73148, 517.47009277],
        ]

        sensors = []
        for i, rx_loc in enumerate(rx_locs):
            rx = build_pcl_receiver(
                rx_id=i,
                point=Point(
                    lat=rx_loc[0],
                    lon=rx_loc[1],
                    alt=rx_loc[2],
                ),
            )
            for j, tx in enumerate(txs):
                sensors.append(
                    PclSensor(
                        id=2 + i * len(rx_locs) + j,
                        transmitter=tx,
                        receiver=rx,
                        error_model=PclMeasurementModel(),
                    )
                )
        return sensors

    def _get_blue_controller(self) -> Controller:
        blue_sensor = build_flores_monostatic_radar(
            Point(
                lat=POSITIONS_OF_INTEREST["Uetliberg"]["lat"],
                lon=POSITIONS_OF_INTEREST["Uetliberg"]["lon"],
                alt=POSITIONS_OF_INTEREST["Uetliberg"]["alt"],
            ),
            1,
            2,
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
        rx_ids = set([s.receiver.id for s in sensors])
        tx_ids = set([s.transmitter.id for s in sensors])
        rx_target_ids = {rx_id: 3 + i for i, rx_id in enumerate(rx_ids)}
        tx_target_ids = {
            tx_id: 3 + len(rx_target_ids) + i for i, tx_id in enumerate(tx_ids)
        }
        used_receiver_ids = []
        used_transmitter_ids = []
        for sensor in sensors:
            receiver_owned = sensor.receiver.id in used_receiver_ids
            transmitter_owned = sensor.transmitter.id in used_transmitter_ids
            c = PclSensorController(
                target_id_rx=rx_target_ids[sensor.receiver.id],
                target_id_tx=tx_target_ids[sensor.transmitter.id],
                sensor=sensor,
                is_blue=True,
                rcs_model=ConstantRcsModel(rcs=1.0),
                own_receiver=not receiver_owned,
                own_transmitter=not transmitter_owned,
            )
            if not receiver_owned:
                used_receiver_ids.append(sensor.receiver.id)
            if not transmitter_owned:
                used_transmitter_ids.append(sensor.transmitter.id)
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
