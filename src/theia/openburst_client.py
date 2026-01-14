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

from theia.types import Radar, RadioClimate, Target


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
    horiz_diagr_att: int = 0
    vert_diagr_att: int = 0
    gain: int = 0
    losses: int = 0
    temp_sys: int = 300
    update_time: int = -1
    txcallsigns: str | None = None
    x: float | None = None
    y: float | None = None


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
    power: float


class LatLonHeightGrid(pydantic.BaseModel):
    lat_start: float
    lat_stop: float
    lat_res: float

    lon_start: float
    lon_stop: float
    lon_res: float

    height_start: float
    height_stop: float
    height_res: float

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # Correct stop values to match the resolution.
        n_points_lat = int(np.ceil((self.lat_stop - self.lat_start) / self.lat_res)) + 1
        n_points_lon = int(np.ceil((self.lon_stop - self.lon_start) / self.lon_res)) + 1
        n_points_height = (
            int(np.ceil((self.height_stop - self.height_start) / self.height_res)) + 1
        )

        self.lat_stop = self.lat_start + (n_points_lat - 1) * self.lat_res
        self.lon_stop = self.lon_start + (n_points_lon - 1) * self.lon_res
        self.height_stop = self.height_start + (n_points_height - 1) * self.height_res

    @property
    def n_points_lat(self) -> int:
        return int(np.round((self.lat_stop - self.lat_start) / self.lat_res)) + 1

    @property
    def n_points_lon(self) -> int:
        return int(np.round((self.lon_stop - self.lon_start) / self.lon_res)) + 1

    @property
    def n_points_height(self) -> int:
        return (
            int(np.round((self.height_stop - self.height_start) / self.height_res)) + 1
        )

    @property
    def n_points(self) -> tuple[int, int, int]:
        return (self.n_points_lat, self.n_points_lon, self.n_points_height)

    def to_openburst(self) -> dict[str, float]:
        return {
            "lat_start": self.lat_start,
            "lat_stop": self.lat_stop,
            "lon_start": self.lon_start,
            "lon_stop": self.lon_stop,
            "min_x": self.lon_start,
            "max_x": self.lon_stop,
            "min_y": self.lat_start,
            "max_y": self.lat_stop,
            "min_z": self.height_start,
            "max_z": self.height_stop,
            "res_x": self.lon_res,
            "res_y": self.lat_res,
            "res_z": self.height_res,
            "amt_pts_x": self.n_points_lon,
            "amt_pts_y": self.n_points_lat,
            "amt_pts_z": self.n_points_height,
        }

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.lat_stop + self.lat_start),
            0.5 * (self.lon_stop + self.lon_start),
            0.5 * (self.height_stop + self.height_start),
        )

    @property
    def points(self) -> np.ndarray:
        n = self.n_points[0] * self.n_points[1] * self.n_points[2]
        points = np.empty((n, 3), dtype=np.float32)
        i = 0
        for lat in np.linspace(self.lat_start, self.lat_stop, self.n_points_lat):
            for lon in np.linspace(self.lon_start, self.lon_stop, self.n_points_lon):
                for height in np.linspace(
                    self.height_start, self.height_stop, self.n_points_height
                ):
                    points[i, :] = lat, lon, height
                    i += 1
        return points


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
    ):
        self._url_radterrain = f"ws://{base_url}:{radterrain_port}/dem"
        self._url_geoplot = f"ws://{base_url}:{geoplot_port}/geoplot"
        self._url_pcl = f"ws://{base_url}:{pcl_port}/pcl"
        self._url_active_detection = f"http://{base_url}:{util_port}/rad_detection"

    def calculate_coverage(
        self,
        radar: Radar,
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
            radar.id,
            radar.lat,
            radar.lon,
            target_flight_height,
            radar.power,
            radar.diameter,
            radar.frequency / 1000.0,  # convert MHz -> GHz
            radar.pulse_width,
            radar.cpi_pulses,
            radar.bandwidth,
            radar.pfa,
            target_cross_section,
            radar.min_elevation,
            radar.max_elevation,
            int(enable_propagation_model),
            int(enable_magl),
        ]

        # Calculate radar coverage.
        with connect(self._url_radterrain) as ws:
            ws.send(",".join([str(p) for p in properties]))
            answer = ws.recv()

        # Convert radar coverage to KML.
        with connect(self._url_geoplot) as ws:
            ws.send(
                json.dumps(
                    {
                        "request_type": "activeCoveragePoints",
                        "nbr_args": 1,
                        "args": ['"' + answer + '"'],
                    }
                )
            )
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
            ws.send(
                json.dumps(
                    {
                        "request_type": "findTxForRx_all",
                        "nbr_args": 2,
                        "args": [
                            [r.model_dump_json() for r in receivers],
                            take_non_los_emitters,
                        ],
                    }
                ).encode("utf-8")
            )
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
        pcl_receivers: list[PclReceiver],
        pcl_emitters: list[PclEmitter],
        grid: LatLonHeightGrid,
        snr_threshold: float = 15.0,
        delay_threshold: float = 1.0,
        integration_time: float = 0.1,
    ) -> list[np.ndarray]:
        with connect(self._url_pcl, ping_timeout=None) as ws:
            ws.send(
                json.dumps(
                    {
                        "request_type": "calcMinDetRCScoverage",
                        "nbr_args": 8,
                        "args": [
                            [r.model_dump_json() for r in pcl_receivers],
                            [e.model_dump_json() for e in pcl_emitters],
                            str(snr_threshold),
                            str(delay_threshold),
                            str(integration_time),
                            json.dumps(grid.to_openburst()),
                            "0",
                            "",
                        ],
                    }
                ).encode("utf-8")
            )
            answer = ws.recv()

        cubes_all_receivers = json.loads(json.loads(answer)["args"][0])
        assert len(cubes_all_receivers) == len(pcl_receivers)
        cubes_all_receivers = [
            np.asarray(json.loads(cube)) for cube in cubes_all_receivers
        ]

        for cube in cubes_all_receivers:
            assert cube.shape == grid.n_points

        return cubes_all_receivers

    def calculate_propagation(
        self,
        transmitter: Radar,
        receiver: Radar,
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
                transmitter.lat,
                -transmitter.lon,
                # convert to feet
                transmitter.antenna_height * METER_TO_FEET,
                receiver.lat,
                -receiver.lon,
                # convert to feet
                receiver.antenna_height * METER_TO_FEET,
                earth_dielectric_constant,
                earth_conductivitiy,
                atmospheric_bending_constant,
                transmitter.frequency,
                radio_climate.value,
                transmitter.polarization.value,
                ground_clutter * METER_TO_FEET,
                0,
                int(oitm),
                transmitter.erp,
            ]
            ws.send(",".join([str(p) for p in params]))
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
        self, radar: Radar, target: Target, doppler_shift_threshold: float = 5.0
    ) -> float:
        radar = radar.model_copy()
        # Convert MHz to GHz.
        radar.frequency /= 1000.0
        response = requests.get(
            self._url_active_detection,
            params={
                "radar": radar_to_openburst_json(radar),
                "target": target_to_openburst_json(target),
                "doppler_shift_threshold": doppler_shift_threshold,
            },
        )

        return float(response.text)


def radar_to_openburst_json(radar: Radar) -> str:
    r = radar.model_dump()
    r["polarization"] = r["polarization"].value
    r["lat"] = r["point"]["lat"]
    r["lon"] = r["point"]["lon"]
    r["alt"] = r["point"]["alt"]
    del r["point"]
    return str(r).replace("'", '"')


def target_to_openburst_json(target: Target) -> str:
    return str(
        {
            "lat": target.lat,
            "lon": target.lon,
            "alt": target.alt,
            "cross_section": target.cross_section,
            "vlon": target.vlon,
            "vlat": target.vlat,
            "vz": target.vz,
        }
    ).replace("'", '"')
