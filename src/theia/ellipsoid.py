from functools import cached_property

import numpy as np
import pydantic


class Ellipsoid(pydantic.BaseModel):
    p1: tuple[float, float, float]
    """Focal point 1 in Cartesian ECEF coordinates [m]"""
    p2: tuple[float, float, float]
    """Focal point 2 in Cartesian ECEF coordinates [m]"""
    r: float
    """Radius of the ellipsoid (string length in pins-and-string construction) [m]"""

    @cached_property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.p1[0] + self.p2[0]) / 2.0,
            (self.p1[1] + self.p2[1]) / 2.0,
            (self.p1[2] + self.p2[2]) / 2.0,
        )

    @cached_property
    def axes_directions(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Orthonormal main semi-axes of the ellipsoid."""
        ax1 = np.array(self.p2) - np.array(self.p1)
        ax1 = ax1 / np.linalg.norm(ax1)

        ax2 = np.array((1.0, 0.0, 0.0))
        if np.abs(np.dot(ax1, ax2)) >= 0.9:
            # Avoid collinear axes.
            ax2 = np.array((0.0, 1.0, 0.0))
        # Subtract non-orthogonal parts.
        ax2 = ax2 - np.dot(ax1, ax2) * ax1
        ax2 = ax2 / np.linalg.norm(ax2)

        ax3 = np.cross(ax1, ax2)
        ax3 = ax3 / np.linalg.norm(ax3)
        return tuple(ax1), tuple(ax2), tuple(ax3)

    @cached_property
    def axes_lengths(self) -> tuple[float, float, float]:
        """Lengths of the ellipsoid's main semi-axes [m]"""
        a = self.r / 2.0
        d = np.sqrt(
            np.square(self.p1[0] - self.p2[0])
            + np.square(self.p1[1] - self.p2[1])
            + np.square(self.p1[2] - self.p2[2])
        )
        b = 0.5 * np.sqrt(np.square(self.r) - np.square(d))
        return (a, b, b)

    def point_on_ellipsoid(self, p: tuple, tol=1e-6) -> bool:
        axes = np.array(self.axes_directions)  # 3x3 orthonormal matrix
        local = axes @ (np.array(p) - np.array(self.center))
        lengths = np.array(self.axes_lengths)
        return abs(np.sum((local / lengths) ** 2) - 1.0) < tol

    def sample_surface_uniformly(
        self,
        rng: np.random.Generator,
    ) -> tuple[float, float, float]:
        """
        Sample a point uniformly on the ellipsoid's surface.

        This means each area element has the same expected number of particles
        to be sampled.
        """
        a, b, c = self.axes_lengths
        # Maximum possible weight (numerical upper bound)
        # w(φ,θ) = sinφ · sqrt(b²c²sin²φcos²θ + a²c²sin²φsin²θ + a²b²cos²φ)
        # Upper-bound: sqrt(max(b²c², a²c², a²b²)) * sinφ ≤ max_semiaxis²
        w_max = np.sqrt(max((b * c) ** 2, (a * c) ** 2, (a * b) ** 2))
        w_max = max(a * b, a * c, b * c)

        while True:
            phi = rng.uniform(0, np.pi)
            theta = rng.uniform(0, 2 * np.pi)

            w = _area_weight(phi, theta, a, b, c)
            assert w <= w_max
            u = rng.uniform(0, w_max)
            if u <= w:
                x = a * np.sin(phi) * np.cos(theta)
                y = b * np.sin(phi) * np.sin(theta)
                z = c * np.cos(phi)

                ax1, ax2, ax3 = self.axes_directions
                return (
                    x * ax1[0] + y * ax2[0] + z * ax3[0] + self.center[0],
                    x * ax1[1] + y * ax2[1] + z * ax3[1] + self.center[1],
                    x * ax1[2] + y * ax2[2] + z * ax3[2] + self.center[2],
                )

    def calculate_radius(self, p: tuple[float, float, float]) -> float:
        p = np.array(p)
        d1 = np.linalg.norm(np.array(self.p1) - p)
        d2 = np.linalg.norm(np.array(self.p2) - p)
        return d1 + d2


class EllipsoidIntersection(pydantic.BaseModel):
    e1: Ellipsoid
    e2: Ellipsoid
    sigma_r1: float
    """Uncertainty in the range of the first ellipsoid"""
    sigma_r2: float
    """Uncertainty in the range of the second ellipsoid"""

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        # Apply rejection sampling: Suggest using distribution of ellipsoid1,
        # then reject based on distribution of ellipsoid2.
        while True:
            p = self.e1.sample_surface_uniformly(rng)
            r = self.e2.calculate_radius(p)
            if self.sigma_r2 > 0:
                prob = (
                    1
                    / np.sqrt(2 * np.pi * self.sigma_r2)
                    * np.exp(-((self.e2.r - r) ** 2) / (2 * self.sigma_r2**2))
                )
            elif np.isclose(r, self.e2.r):
                prob = 1.0
            else:
                prob = 0.0

            u = rng.uniform()
            if u <= prob:
                return p


def _area_weight(
    phi: np.ndarray, theta: np.ndarray, a: float, b: float, c: float
) -> np.ndarray:
    """
    Surface area element (without dθdφ):
        w(θ,φ) = sinθ · sqrt(b²c²sin²θcos²φ + a²c²sin²θsin²φ + a²b²cos²θ)

    This is |r_θ x r_φ|, the Jacobian of the parametrisation.

    Notes
    -----
    Source: Equ. (26) in https://mathworld.wolfram.com/Ellipsoid.html
    """
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    inner = (
        (b * c) ** 2 * sin_p**2 * cos_t**2
        + (a * c) ** 2 * sin_p**2 * sin_t**2
        + (a * b) ** 2 * cos_p**2
    )
    return sin_p * np.sqrt(inner)
