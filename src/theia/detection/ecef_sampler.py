import datetime

import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.updater.kalman import LinearGaussian

from theia.coordinates import (
    CoordinateTransformations,
    calculate_azimuth_angle,
    calculate_elevation_angle,
)
from theia.distance import line_of_sight_distance
from theia.ellipsoid import Ellipsoid, EllipsoidIntersection
from theia.types import MonostaticRadarDetection, PclDetection, PetDetection, Point
from theia.util import normal_pdf


class EcefDetectionSampler:
    """
    Sample Cartesian ECEF positions from detection information.
    """

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def _sample_position(
        self,
        target_position: Point,
        sigmas: tuple[float, float, float],
        detection_time: datetime.datetime,
    ) -> Detection:
        mean = CoordinateTransformations.geodetic_to_cartesian(
            target_position.lat,
            target_position.lon,
            target_position.alt,
        )
        # Sample a fake detection.
        x, y, z = self._rng.normal(mean, sigmas)
        measured_state = np.array([x, y, z])

        covar = np.diag(sigmas)

        return Detection(
            measured_state,
            measurement_model=LinearGaussian(
                ndim_state=6,
                mapping=(0, 2, 4),
                noise_covar=covar,
            ),
            timestamp=detection_time,
        )

    def sample_monostatic(self, detection: MonostaticRadarDetection) -> Detection:
        sigma = np.max(
            (
                detection.sigma_target_range,
                detection.sigma_azimuth * detection.target_range,
                detection.sigma_elevation * detection.target_range,
            )
        )
        return self._sample_position(
            detection.target.point,
            (sigma, sigma, sigma),
            detection.time,
        )

    def sample_pcl(self, detections: list[PclDetection]) -> Detection:
        sigma = np.max([d.sigma_bistatic_range for d in detections])
        # All detections share the same target, i. e. we can select
        # any of them to access the true target state.
        detection = detections[0]
        return self._sample_position(
            detection.target.point,
            (sigma, sigma, sigma),
            detection.time,
        )

    def sample_two_pcl_detections(
        self,
        detections: tuple[PclDetection, PclDetection],
        n_samples: int = 32,
    ) -> Detection:
        detection = detections[0]
        p11 = CoordinateTransformations.geodetic_to_cartesian(
            *detection.sensor.receiver.point.as_tuple()
        )
        p12 = CoordinateTransformations.geodetic_to_cartesian(
            *detection.sensor.transmitter.point.as_tuple()
        )
        d = np.linalg.norm(np.array(p11) - np.array(p12))
        e1 = Ellipsoid(p1=p11, p2=p12, r=detection.bistatic_range + d)

        detection = detections[1]
        p21 = CoordinateTransformations.geodetic_to_cartesian(
            *detection.sensor.receiver.point.as_tuple()
        )
        p22 = CoordinateTransformations.geodetic_to_cartesian(
            *detection.sensor.transmitter.point.as_tuple()
        )
        d = np.linalg.norm(np.array(p21) - np.array(p22))
        e2 = Ellipsoid(p1=p21, p2=p22, r=detection.bistatic_range + d)

        intersection = EllipsoidIntersection(
            e1=e1,
            e2=e2,
            sigma_r1=detections[0].sigma_bistatic_range,
            sigma_r2=detections[1].sigma_bistatic_range,
        )
        points = np.array([intersection.sample(self._rng) for _ in range(n_samples)])

        covar = np.diag(np.std(points, axis=0))

        return Detection(
            np.mean(points, axis=0),
            measurement_model=LinearGaussian(
                ndim_state=6, mapping=(0, 2, 4), noise_covar=covar
            ),
            timestamp=detections[0].time,
        )

    def sample_multiple_pet(self, detections: list[PetDetection]) -> Detection:
        if len(detections) < 2:
            raise ValueError(
                "Need at least 2 PET detections to estimate target position in ECEF space"
            )
        sigmas = []
        for detection in detections:
            d = line_of_sight_distance(
                detection.target.lat,
                detection.target.lon,
                detection.target.alt,
                detection.radar.receiver.lat,
                detection.radar.receiver.lon,
                detection.radar.receiver.alt,
            )
            sigmas.append(d * detection.azimuth)
            sigmas.append(d * detection.elevation)
        sigma = np.max(sigmas)

        # All detections share the same target, i. e. we can select
        # any of them to access the true target state.
        detection = detections[0]
        return self._sample_position(
            detection.target,
            (sigma, sigma, sigma),
            detection.time,
        )

    def sample_two_pcl_one_pet_detections_to_ecef(
        self,
        pcl_detections: tuple[PclDetection, PclDetection],
        pet_detection: PetDetection,
    ) -> Detection:
        """
        Sample a position in ECEF space around the ground truth one using
        error propagation.

        Parameters
        ----------
        pcl_detections: tuple[PclDetection, PclDetection]
            PCL detections
        pet_detection: PetDetection
            PET detection

        Returns
        -------
        stonesoup.types.detection.Detection
            A detection representing a position in 3D space, including measurement model

        Raises
        ------
        ValueError
            If `len(detections) < 2`
        """
        d = line_of_sight_distance(
            pet_detection.target.lat,
            pet_detection.target.lon,
            pet_detection.target.alt,
            pet_detection.radar.receiver.lat,
            pet_detection.radar.receiver.lon,
            pet_detection.radar.receiver.alt,
        )
        sigma = np.max(
            (
                d * pet_detection.azimuth,
                d * pet_detection.elevation,
                pcl_detections[0].sigma_bistatic_range,
                pcl_detections[1].sigma_bistatic_range,
            )
        )

        # All detections share the same target, i. e. we can select
        # any of them to access the true target state.
        detection = pet_detection
        return self._sample_position(
            detection.target,
            (sigma, sigma, sigma),
            detection.time,
        )

    def sample_one_pcl_one_pet_detections_to_ecef(
        self,
        pcl_detection: PclDetection,
        pet_detection: PetDetection,
        n_samples: int = 32,
    ) -> Detection:
        p1 = CoordinateTransformations.geodetic_to_cartesian(
            *pcl_detection.sensor.receiver.point.as_tuple()
        )
        p2 = CoordinateTransformations.geodetic_to_cartesian(
            *pcl_detection.sensor.transmitter.point.as_tuple()
        )
        e = Ellipsoid(
            p1=p1,
            p2=p2,
            r=pcl_detection.bistatic_range + np.linalg.norm(p1 - p2),
        )

        def sample_point() -> tuple[float, float, float]:
            while True:
                x = e.sample_surface_uniformly(self._rng)
                p = CoordinateTransformations.cartesian_to_geodetic(*x)
                p = Point(
                    lat=p[0],
                    lon=p[1],
                    alt=p[2],
                )
                elevation_angle = (
                    calculate_elevation_angle(
                        pet_detection.sensor.transmitter.point,
                        p,
                    ),
                )
                azimuth_angle = (
                    calculate_azimuth_angle(
                        pet_detection.sensor.transmitter.point,
                        p,
                    ),
                )

                weight = normal_pdf(
                    pet_detection.azimuth,
                    pet_detection.sigma_azimuth,
                    azimuth_angle,
                ) * normal_pdf(
                    pet_detection.elevation,
                    pet_detection.sigma_elevation,
                    elevation_angle,
                )

                u = self._rng.uniform()
                if u <= weight:
                    return x

        points = np.array([sample_point(self._rng) for _ in range(n_samples)])
        covar = np.diag(np.std(points, axis=0))

        return Detection(
            np.mean(points, axis=0),
            measurement_model=LinearGaussian(
                ndim_state=6, mapping=(0, 2, 4), noise_covar=covar
            ),
            timestamp=pcl_detection.time,
        )

    def sample(
        self,
        monostatic_detections: list[MonostaticRadarDetection],
        pcl_detections: list[PclDetection],
        pet_detections: list[PetDetection],
        track_exists: bool,
    ) -> Detection | None:
        detection: Detection | None = None
        if len(monostatic_detections) > 0:
            # Whenever we have monostatic detections, we use those detections
            # to initialize or update the track.
            # Select detection with shortest range and initiate or update
            # the track.
            detection = sorted(monostatic_detections, key=lambda d: d.target_range)[0]
            detection = self.sample_monostatic(detection)
        elif len(pcl_detections) >= 3:
            detection = self.sample_pcl(pcl_detections)
        elif len(pet_detections) >= 2:
            detection = self.sample_multiple_pet(pet_detections)
        elif len(pcl_detections) == 2 and len(pet_detections) == 1:
            detection = self.sample_two_pcl_one_pet_detections_to_ecef(
                tuple(pcl_detections),
                pet_detections[0],
            )
        elif (len(pcl_detections) == 2) and track_exists:
            detection = self.sample_two_pcl_detections(tuple(pcl_detections))
        elif (len(pcl_detections) == 1) and (len(pet_detections) == 1) and track_exists:
            detection = self.sample_one_pcl_one_pet_detections_to_ecef(
                pcl_detections[0],
                pet_detections[0],
            )
        return detection
