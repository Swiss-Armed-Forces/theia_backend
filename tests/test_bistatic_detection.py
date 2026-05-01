from pathlib import Path
import unittest

import json

import numpy as np

from theia.data_loading import load_bakom_ukw_transmitters
from theia.detection.pcl import PclDetector
from theia.distance import line_of_sight_distance
from theia.grids import LatLonHeightGrid
from theia.openburst_client import OpenburstClient
from theia.test_data import get_uetliberg_radar
from theia.types import Receiver, Transmitter


PATH = (
    f"{(Path(__file__).parent / 'test_data' / 'pcl_coverage' / 'file.json').absolute()}"
)


class PclMinDetectableRcsTest(unittest.TestCase):
    def test_min_detectable_rcs(self):
        rxs, txs, grids, values_loaded = load_pcl_test_data(PATH)
        for rx, tx, grid, true_values in list(
            zip(rxs, txs, grids, values_loaded, strict=True)
        )[:-1]:
            detector = PclDetector()
            result_calc = detector.minimum_detectable_rcs_vector(rx, tx, grid.points)
            result_calc = result_calc.reshape(grid.n_points)
            np.nan_to_num(result_calc, copy=False, nan=-1)

            error = np.abs((true_values - np.clip(result_calc, -1, 150.0)))
            for i, e in enumerate(error.flatten()):
                if result_calc.flatten()[i] == -1:
                #     # An error was raised. We assume that this was due to 
                #     # terrain obstructing the line-of-sight.
                #     # Unfortunately, openBURST has a bug that prevents
                #     # line-of-sight checks, so we have to exclude these points
                #     # from the tests until openBURST is fixed.
                #     # TODO Update after fix of issue:
                #     # https://github.com/Swiss-Armed-Forces/openburst/issues/29
                    continue
                self.assertLess(e, 0.3)


def save_pcl_test_data(
    path: str,
    data: tuple[
        list[Receiver],
        list[Transmitter],
        LatLonHeightGrid,
        list[np.ndarray],
    ]
    | None = None,
):
    if data is None:
        p = get_uetliberg_radar(integration_time=0.1).receiver.point

        transmitters = load_bakom_ukw_transmitters()
        txs = list(
            filter(
                lambda t: (
                    line_of_sight_distance(*p.as_tuple(), *t.point.as_tuple()) < 50_000
                ),
                transmitters,
            )
        )[:3]

        grid = LatLonHeightGrid(
            lat_start=46.9973,
            lat_stop=47.5282,
            lat_res=(47.5282 - 46.9973) / 4.0,
            lon_start=7.8635,
            lon_stop=9.1956,
            lon_res=(9.1956 - 7.8635) / 4.0,
            height_start=1000.0,
            height_stop=1000.0,
            height_res=1000.0,
        )

        rxs: list[Receiver] = []
        true_values: list[list[float]] = []
        for tx in txs:
            rx = get_uetliberg_radar(
                integration_time=0.1,
                tx_bandwidth=tx.bandwidth,
            ).receiver
            # openBURST ignores receiver noise figure in PCL SNR calculation.
            rx.noise_figure = 0.0
            rxs.append(rx)
            true_values.append(np.full(grid.n_points, np.nan).tolist())
    else:
        rxs, txs, grid, true_values = data
        true_values = [values.tolist() for values in true_values]

    configs = []
    for rx, tx, values in zip(rxs, txs, true_values, strict=True):
        config = {
            "rx": rx.model_dump(mode="json"),
            "tx": tx.model_dump(mode="json"),
            "grid": grid.model_dump(mode="json"),
            "true_values": values,
        }
        configs.append(config)

    with open(path, "w") as file:
        json.dump(configs, file)


def load_pcl_test_data(
    path: str,
) -> tuple[list[Receiver], list[Transmitter], list[LatLonHeightGrid], list[np.ndarray]]:
    with open(path, "r") as file:
        configs_loaded = json.load(file)
    rxs_loaded = [Receiver.model_validate(entry["rx"]) for entry in configs_loaded]
    txs_loaded = [Transmitter.model_validate(entry["tx"]) for entry in configs_loaded]
    grids = [LatLonHeightGrid.model_validate(entry["grid"]) for entry in configs_loaded]
    true_values = [np.array(entry["true_values"]) for entry in configs_loaded]
    return rxs_loaded, txs_loaded, grids, true_values


def calculate_reference_min_rcs(path: str):
    """
    Calculate reference data for PCL minimum RCS calculation.

    An openBURST server needs to be running at localhost in order to calculate
    the values.
    """
    save_pcl_test_data(path)
    rxs, txs, grids, _ = load_pcl_test_data(path)

    client = OpenburstClient("localhost")

    true_values = []
    for rx, tx, grid in zip(rxs, txs, grids, strict=True):
        result = client.calculate_min_det_rcs_coverage(
            rx,
            tx,
            grid,
        )[0]
        true_values.append(result)

    save_pcl_test_data(path, data=(rxs, txs, grid, true_values))


if __name__ == "__main__":
    # Uncomment the following line and comment the unittest.main() call to
    # regenerate the test data.

    # calculate_reference_min_rcs(
    #     PATH
    # )

    unittest.main()
