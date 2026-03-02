import numpy as np
from stonesoup.base import Property
from stonesoup.functions import jacobian as approx_jacobian
from stonesoup.models.base import ReversibleModel
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement
from stonesoup.types.state import State, StateVector, StateVectors
from stonesoup.types.detection import Clutter, Detection

from theia.measurement import MonostaticMeasurementTransformations
from theia.types import CLUTTER_TARGET, MonostaticRadarDetection, Point


class MonostaticEcefMeasurement(NonLinearGaussianMeasurement, ReversibleModel):
    p_radar: Point = Property(doc="Radar position")

    @property
    def ndim_meas(self) -> int:
        return 3

    def function(
        self,
        state: State,
        noise: bool = False,
        **kwargs,
    ) -> StateVector:
        if noise:
            raise NotImplementedError()
        N = state.state_vector.shape[1]
        result = np.zeros((3, N))
        for i in range(N):
            ecef_pos = state.state_vector[self.mapping, i]
            elevation, azimuth, range_m = (
                MonostaticMeasurementTransformations.cartesian_to_elevation_azimuth_range(
                    self.p_radar, tuple(ecef_pos.flatten())
                )
            )
            result[:, i] = (float(elevation), float(azimuth), float(range_m))

        if N == 1:
            return StateVector(result)  # (3, 1)
        else:
            return StateVectors(result)  # (3, N)

    def inverse_function(
        self,
        detection: Detection,
        **kwargs,
    ) -> StateVector:
        elevation: float = detection.state_vector[0, 0]
        azimuth: float = detection.state_vector[1, 0]
        range_m: float = detection.state_vector[2, 0]

        p_ecef = (
            MonostaticMeasurementTransformations.elevation_azimuth_range_to_cartesian(
                self.p_radar,
                elevation,
                azimuth,
                range_m,
            )
        )

        result = StateVector(np.zeros((self.ndim_state, 1)))
        result[self.mapping, :] = np.array(p_ecef, dtype=np.float64).reshape(3, 1)
        return result

    def jacobian(self, state: State, **kwargs) -> np.ndarray:
        return approx_jacobian(self.function, state, step_size=1.0)


class MonostaticDetectionFactory:
    @staticmethod
    def from_theia(detection: MonostaticRadarDetection) -> Detection | Clutter:
        cov = np.diag(
            [
                detection.sigma_elevation**2,
                detection.sigma_azimuth**2,
                detection.sigma_target_range**2,
            ]
        )
        model = MonostaticEcefMeasurement(
            p_radar=detection.radar.receiver.point,
            ndim_state=6,
            mapping=(0, 2, 4),
            noise_covar=cov,
        )
        state_vector = [
            detection.elevation_angle,
            detection.azimuth_angle,
            detection.target_range,
        ]
        metadata = {
            "radar_id": detection.radar.receiver.id,
            "target_id": detection.target.id,
            "snr": detection.snr,
        }
        if detection.target.id == CLUTTER_TARGET.id:
            return Clutter(
                state_vector=state_vector,
                timestamp=detection.time,
                metadata=metadata,
                measurement_model=model,
            )
        else:
            return Detection(
                state_vector=state_vector,
                timestamp=detection.time,
                metadata=metadata,
                measurement_model=model,
            )
