import base64
from io import BytesIO
import json
import re
from PIL import Image
from typing import Literal

import geopandas as gpd
import numpy as np
import pydantic
import requests
import shapely
from websockets.sync.client import connect

from theia.coordinates import CoordinateTransformations
from theia.grids import LatLonHeightGrid
from theia.types import Polarization, Sensor, RadioClimate, Receiver, Target, Transmitter
from theia.util import to_dB


def grid_to_openburst(grid: LatLonHeightGrid) -> dict[str, float]:
    return {
        "lat_start": grid.lat_start,
        "lat_stop": grid.lat_stop,
        "lon_start": grid.lon_start,
        "lon_stop": grid.lon_stop,
        "min_x": grid.lon_start,
        "max_x": grid.lon_stop,
        "min_y": grid.lat_start,
        "max_y": grid.lat_stop,
        "min_z": grid.height_start,
        "max_z": grid.height_stop,
        "res_x": grid.lon_res,
        "res_y": grid.lat_res,
        "res_z": grid.height_res,
        "amt_pts_x": grid.n_points_lon,
        "amt_pts_y": grid.n_points_lat,
        "amt_pts_z": grid.n_points_height,
    }


class PclReceiver(pydantic.BaseModel):
    name: str
    rx_id: int
    masl: int
    lat: float
    lon: float
    x: int = 0
    y: int = 0
    ahmagl: int = 15
    signal_type: Literal["FM"] = "FM"
    limit_distance: int = 50000
    lostxids: list = []  # IDs of emitters (Tx) for line-of-sight calculation
    status: int = 1
    txcallsigns: str = ""
    bandwidth: int = 8000
    """Bandwidth [kHz]"""
    horiz_diagr_att: int = 0
    vert_diagr_att: int = 0
    gain: float = 0
    """Gain [dB]"""
    losses: float = 0
    """Losses [dB]"""
    temp_sys: int = 300
    update_time: int = -1


class PclEmitter(pydantic.BaseModel):
    tx_id: int
    callsign: str
    sitename: str
    lat: float
    lon: float
    masl: float
    ahmagl: float
    freq: float
    bandwidth: float
    erp_h: float | Literal["UNDEFINED"]
    erp_v: float | Literal["UNDEFINED"]
    type: Literal["directional", "OMNI"]
    horiz_diagr_att: list[float] | float
    vert_diagr_att: list[float] | float | Literal["UNDEFINED"]
    pol: Literal["H", "V"]
    signal_type: Literal["FM"]
    losrxids: list[int]
    status: Literal[1]


class OpenburstClient:
    _RADAR_COVERAGE: int = 559765
    _ELEVATION: int = 6200901
    _PROPAGATION: int = 29824733

    def __init__(
        self,
        base_url: str,
        radterrain_port: int = 9978,
        geoplot_port: int = 9943,
        pcl_port: int = 7999,
        util_port: int = 3875,
        debugging: bool = False,
    ):
        self._url_radterrain = f"ws://{base_url}:{radterrain_port}/dem"
        self._url_geoplot = f"ws://{base_url}:{geoplot_port}/geoplot"
        self._url_pcl = f"ws://{base_url}:{pcl_port}/pcl"
        self._url_active_detection = f"http://{base_url}:{util_port}/rad_detection"
        self._debugging = debugging

    def calculate_coverage(
        self,
        radar: Sensor,
        target_flight_height: float,
        target_cross_section: float,
        enable_propagation_model: bool = False,
        enable_magl: bool = False,
    ) -> shapely.geometry.polygon.Polygon:
        """
        Calculate radar coverage.

        Returns
        -------
        Shapely polygon representing radar coverage in the WGS84 coordinate system.
        """
        # Build the request message.
        properties = [
            self._RADAR_COVERAGE,
            radar.transmitter.id,
            radar.transmitter.lat,
            radar.transmitter.lon,
            target_flight_height,
            radar.transmitter.power,
            radar.transmitter.antenna_diameter,
            radar.transmitter.frequency / 1000.0,  # convert MHz -> GHz
            radar.transmitter.pulse_width,
            radar.receiver.cpi_pulses,
            radar.transmitter.bandwidth,
            radar.receiver.pfa,
            target_cross_section,
            radar.receiver.min_elevation,
            radar.receiver.max_elevation,
            int(enable_propagation_model),
            int(enable_magl),
        ]

        # Calculate radar coverage.
        with connect(self._url_radterrain) as ws:
            msg = ",".join([str(p) for p in properties])
            if self._debugging:
                print(msg)
            ws.send(msg)
            answer = ws.recv()

        # Convert radar coverage to KML.
        with connect(self._url_geoplot) as ws:
            msg = {
                "request_type": "activeCoveragePoints",
                "nbr_args": 1,
                "args": ['"' + answer + '"'],
            }
            if self._debugging:
                print(msg)
            ws.send(json.dumps(msg))
            answer2 = ws.recv()
            kml = json.loads(json.loads(answer2)["args"][0])

        # Parse KML file.
        with open("test.kml", "w") as file:
            file.write(kml)

        df = gpd.read_file("test.kml")
        polygon = df.iloc[0]["geometry"]

        return polygon

    def elevation_at(self, query_lat: float, query_lon: float) -> float:
        """Query elevation [m] at the query position."""
        # Needed for distance calculation of query point to POI. Dummy for elevation calc.
        poi_lat = 46.25
        poi_lon = 7.12

        query = [self._ELEVATION, query_lat, query_lon, poi_lat, poi_lon]
        query = [str(part) for part in query]

        with connect(self._url_radterrain) as ws:
            ws.send(",".join(query))
            answer = ws.recv()

        height = json.loads(answer)[1]
        return height

    def get_pcl_receivers_emitters(
        self, receivers: list[PclReceiver], take_non_los_emitters: bool = True
    ) -> tuple[list[PclReceiver], list[PclEmitter]]:
        """
        Query close-by emitters for the given receivers.

        Returns
        -------
        receivers: list[PclReceiver]
            Receivers for the PCL. The property lostxids is set to the IDs of
            the close-by emitters.
        emitters: list[PclSender]
            Suggested emitters to be used for PCL.
        """
        with connect(self._url_pcl) as ws:
            msg = json.dumps(
                {
                    "request_type": "findTxForRx_all",
                    "nbr_args": 2,
                    "args": [
                        [r.model_dump_json() for r in receivers],
                        take_non_los_emitters,
                    ],
                }
            ).encode("utf-8")
            if self._debugging:
                print(msg)
            ws.send(msg)
            answer = json.loads(ws.recv())

            receivers_obtained = [
                PclReceiver.model_validate_json(config) for config in answer["args"][0]
            ]
            emitters = [
                PclEmitter.model_validate_json(config) for config in answer["args"][1]
            ]
            for recv in receivers_obtained:
                recv.txcallsigns = ",".join([e.callsign for e in emitters])

            return receivers_obtained, emitters

    def calculate_min_det_rcs_coverage(
        self,
        receiver: Receiver,
        transmitter: Transmitter,
        grid: LatLonHeightGrid,
        snr_threshold: float = 15.0,
        delay_threshold: float = 1.0,
        integration_time: float = 0.1,
    ) -> list[np.ndarray]:
        rx = OpenBurstConverter.receiver_to_pcl_receiver(
            receiver,
            transmitter.frequency,
        )
        tx = OpenBurstConverter.transmitter_to_pcl_emitter(transmitter)
        rx.txcallsigns = tx.callsign
        tx.losrxids = [rx.rx_id]
        with connect(self._url_pcl, ping_timeout=None) as ws:
            msg = json.dumps(
                {
                    "request_type": "calcMinDetRCScoverage",
                    "nbr_args": 8,
                    "args": [
                        [rx.model_dump_json()],
                        [tx.model_dump_json()],
                        str(snr_threshold),
                        str(delay_threshold),
                        str(integration_time),
                        json.dumps(OpenBurstConverter.grid_to_openburst(grid)),
                        "0",
                        "",
                    ],
                }
            ).encode("utf-8")
            if self._debugging:
                print(msg)
            ws.send(msg)
            answer = ws.recv()

        cubes_all_receivers = json.loads(json.loads(answer)["args"][0])
        assert len(cubes_all_receivers) == 1
        cubes_all_receivers = [
            np.asarray(json.loads(cube)) for cube in cubes_all_receivers
        ]

        for cube in cubes_all_receivers:
            assert cube.shape == grid.n_points

        return cubes_all_receivers

    def calculate_propagation(
        self,
        radar: Sensor,
        radio_climate: RadioClimate = RadioClimate.CONTINENTAL_TEMPERATE,
        earth_dielectric_constant: float = 13.0,  # [no units]
        earth_conductivitiy: float = 0.002,  # [S / m]
        atmospheric_bending_constant: float = 301.0,  # [N-units]
        ground_clutter: float = 3.048,  # [m]; default is 10 feet
        oitm: bool = True,  # use original ITM model
    ):
        METER_TO_FEET = 3.28084
        with connect(self._url_radterrain, ping_timeout=None) as ws:
            params = [
                self._PROPAGATION,
                radar.transmitter.lat,
                -radar.transmitter.lon,
                # convert to feet
                radar.transmitter.antenna_height * METER_TO_FEET,
                radar.receiver.lat,
                -radar.receiver.lon,
                # convert to feet
                radar.receiver.antenna_height * METER_TO_FEET,
                earth_dielectric_constant,
                earth_conductivitiy,
                atmospheric_bending_constant,
                radar.transmitter.frequency,
                radio_climate.value,
                radar.transmitter.polarization.value,
                ground_clutter * METER_TO_FEET,
                0,
                int(oitm),
                radar.transmitter.erp,
            ]
            msg = ",".join([str(p) for p in params])
            if self._debugging:
                print(msg)
            ws.send(msg)
        answer = ws.recv()

        # Load the image.
        img_str = json.loads(answer)[1]["height_profile.png"]
        image_bytes = base64.b64decode(img_str)
        img = Image.open(BytesIO(image_bytes))

        # Extract the losses.
        text = json.loads(answer)[1]["TX_-to-RX_.txt"]

        free_space_loss_match = re.search(
            r"Free space path loss:\s*([0-9]+(?:\.[0-9]+)?)\s*dB", text
        )
        longley_rice_loss_match = re.search(
            r"Longley-Rice path loss:\s*([0-9]+(?:\.[0-9]+)?)\s*dB", text
        )
        terrain_shielding_loss_match = re.search(
            r"Attenuation due to terrain shielding:\s*([0-9]+(?:\.[0-9]+)?)\s*dB", text
        )

        free_space_loss = (
            free_space_loss_match.group(1) if free_space_loss_match else np.nan
        )
        longley_rice_loss = (
            longley_rice_loss_match.group(1) if longley_rice_loss_match else np.nan
        )
        terrain_shielding_loss = (
            terrain_shielding_loss_match.group(1)
            if terrain_shielding_loss_match
            else np.nan
        )

        return img, free_space_loss, longley_rice_loss, terrain_shielding_loss

    def active_radar_probability_of_detection(
        self,
        radar: Sensor,
        target: Target,
        rcs: float,
        doppler_shift_threshold: float = 5.0,
    ) -> float:
        radar = radar.model_copy()
        # Convert MHz to GHz.
        params = {
            "radar": radar_to_openburst_json(radar),
            "target": target_to_openburst_json(target, rcs),
            "doppler_shift_threshold": doppler_shift_threshold,
        }
        response = requests.get(
            self._url_active_detection,
            params=params,
        )

        return float(response.text)


def radar_to_openburst_json(radar: Sensor) -> str:
    r = {
        "lat": radar.transmitter.lat,
        "lon": radar.transmitter.lon,
        "alt": radar.transmitter.alt,
        "power": radar.transmitter.power,
        "diameter": radar.transmitter.antenna_diameter,
        "frequency": radar.transmitter.frequency / 1000,  # MHz -> GHz
        "pulse_width": radar.transmitter.pulse_width,
        "cpi_pulses": radar.receiver.cpi_pulses,
        "bandwidth": radar.transmitter.bandwidth,
        "pfa": radar.receiver.pfa,
        "polarization": radar.transmitter.polarization.value,
    }
    return str(r).replace("'", '"')


def target_to_openburst_json(target: Target, rcs: float) -> str:
    vlat, vlon, valt = CoordinateTransformations.velocity_cartesian_to_geodetic(
        target.point,
        target.velocity,
    )
    return str(
        {
            "lat": target.lat,
            "lon": target.lon,
            "alt": target.alt,
            "cross_section": rcs,
            "vlon": float(vlon),
            "vlat": float(vlat),
            "vz": float(valt),
        }
    ).replace("'", '"')


class OpenBurstConverter:
    @staticmethod
    def transmitter_to_pcl_emitter(tx: Transmitter) -> PclEmitter:
        return PclEmitter(
            tx_id=tx.id,
            callsign=str(tx.id),
            sitename=str(tx.id),
            lat=tx.lat,
            lon=tx.lon,
            masl=tx.alt,
            ahmagl=tx.antenna_height,
            freq=tx.frequency,
            bandwidth=tx.bandwidth * 1000,
            erp_h=to_dB(tx.erp) if tx.polarization == Polarization.HORIZONTAL else "UNDEFINED",
            erp_v=to_dB(tx.erp) if tx.polarization == Polarization.VERTICAL else "UNDEFINED",
            type="directional",
            horiz_diagr_att=tx.horizontal_attenuation.attenuation_table_values
            if tx.horizontal_attenuation is not None
            else 0.0,
            vert_diagr_att=tx.vertical_attenuation.attenuation_table_values
            if tx.vertical_attenuation is not None
            else 0.0,
            pol="H" if tx.polarization == Polarization.HORIZONTAL else "V",
            signal_type="FM",
            losrxids=[],
            status=1,
        )

    @staticmethod
    def receiver_to_pcl_receiver(rx: Receiver, frequency_Mz: float) -> PclReceiver:
        return PclReceiver(
            name=str(rx.id),
            rx_id=rx.id,
            masl=int(np.round(rx.alt)),
            lat=rx.lat,
            lon=rx.lon,
            ahmagl=rx.antenna_height,
            signal_type="FM",
            bandwidth=rx.bandwidth * 1000.0,
            gain=rx.antenna_gain(frequency_Mz),
        )

    @staticmethod
    def grid_to_openburst(grid: LatLonHeightGrid) -> dict[str, float]:
        return {
            "lat_start": grid.lat_start,
            "lat_stop": grid.lat_stop,
            "lon_start": grid.lon_start,
            "lon_stop": grid.lon_stop,
            "min_x": grid.lon_start,
            "max_x": grid.lon_stop,
            "min_y": grid.lat_start,
            "max_y": grid.lat_stop,
            "min_z": grid.height_start,
            "max_z": grid.height_stop,
            "res_x": grid.lon_res,
            "res_y": grid.lat_res,
            "res_z": grid.height_res,
            "amt_pts_x": grid.n_points_lon,
            "amt_pts_y": grid.n_points_lat,
            "amt_pts_z": grid.n_points_height,
        }
