import itertools
import numpy as np


def _standard_transformation(
    A: np.typing.NDArray, B: np.typing.NDArray
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """
    Construct transformation to align ellipsis with focal points A, B with x-axis
    and have the origin at their middle point.

    The transformation is as follows: ``x_new = R @ (x + translation_vector)``.

    Parameters
    ----------
    A: np.typing.NDArray
        Focus point
    B: np.typing.NDArray
        Other focus point

    Returns
    -------
    translation_vector: np.NDArray
    R: np.typing.NDArray
        Rotation matrix
    """
    # Step 1: Translate to origin
    midpoint = (A + B) / 2
    A1 = A - midpoint
    B1 = B - midpoint

    # Step 2: Rotation
    v = B1 - A1  # or just B - A
    v_norm = v / np.linalg.norm(v)
    x_axis = np.array([1, 0, 0])

    # Check if already aligned
    dot = np.dot(v_norm, x_axis)
    if np.abs(dot - 1) < 1e-10:
        # Already aligned
        R = np.eye(3)
    elif np.abs(dot + 1) < 1e-10:
        # Anti-aligned, rotate 180° around y or z
        R = np.diag([1, -1, -1])
    else:
        # General rotation
        k = np.cross(v_norm, x_axis)
        k = k / np.linalg.norm(k)
        theta = np.arccos(dot)

        # Rodrigues formula
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return -midpoint, R


class Ellipsoid:
    def __init__(
        self,
        p1: np.typing.ArrayLike,
        p2: np.typing.ArrayLike,
        bistatic_range: float,
    ):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        assert p1.shape == (3,) and p2.shape == (3,)
        d2: float = (
            np.square(p1[0] - p2[0])
            + np.square(p1[1] - p2[1])
            + np.square(p1[2] - p2[2])
        )

        self._p1 = p1
        self._p2 = p2
        self._bistatic_range = bistatic_range

        if d2 > np.square(bistatic_range):
            raise ValueError(
                f"Distance between transmitter and receiver is greater than bistatic range! ({d2:.1} > {bistatic_range**2:.1})"
            )

        # Determine ellipsoid parameters.
        self._a: float = 0.5 * bistatic_range
        self._b: float = 0.5 * np.sqrt(np.square(bistatic_range) - d2)

        # Determine transformation.
        self._t, self._R = _standard_transformation(p1, p2)

    def _transform_world_to_standard(
        self, p: np.typing.ArrayLike
    ) -> np.typing.ArrayLike:
        return self._R @ (p + self._t)

    def _transform_standard_to_world(
        self, p: np.typing.ArrayLike
    ) -> np.typing.ArrayLike:
        return self._R.T @ p - self._t

    def is_inside(self, point: tuple[float, float, float]) -> bool:
        p = self._transform_world_to_standard(point)
        return (
            np.square(p[0] / self._a)
            + np.square(p[1] / self._b)
            + np.square(p[2] / self._b)
        ) <= 1.0

    def is_on_surface(
        self,
        point: tuple[float, float, float],
        tol: float = 1e-2,
        point_in_world_coord: bool = True,
    ) -> bool:
        r"""
        Test whether the point in Cartesian coordinates lies approximately on
        the ellipsoid's surface.

        The ``tol`` parameter controls how much relative deviation is still accepted
        (after transformation so that the major axis lies on the x-axis and is
        centered at the origin):

        .. math::

            \frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{b^2} - 1 \leq tol.

        Parameters
        ----------
        point: tuple[float, float, float]
            Point in Cartesian coordinates for which to check whether it lies on
            the ellipsoid surface.
        tol: float, default 1e-2
            Relative tolerance (see :ref:`notes`)
        point_in_world_coord: bool, default True
            Whether the point is given in world coordinates or in standardised
            (major axis x-axis aligned, centered at origin) coordinates.
            Most probably only needed for debugging.

        .. _notes:

        Notes
        -----
        The parameter ``tol`` indicates the tolerance in distance test!
        In the special case of a sphere (``r = a = b``), the tolerance criterion
        for x to lie on the sphere with radius ``r`` is

        .. math::

            \frac{x}{r}^2 + \frac{y}{r}^2 - 1 \leq tol

        Therefore, a tolerance value of ``0.01`` means that the point's
        corresponding radius deviates less than 1% from the sphere's radius.
        """
        p = self._transform_world_to_standard(point) if point_in_world_coord else point

        return (
            np.abs(
                np.square(p[0] / self._a)
                + np.square(p[1] / self._b)
                + np.square(p[2] / self._b)
                - 1.0
            )
            <= tol
        )

    def sample_surface(
        self,
        n_theta: int = 50,
        n_phi: int = 100,
    ) -> list[np.typing.NDArray[np.float64]]:
        """Sample surface using spherical parameterization"""
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)

        # Parametric equations for ellipsoid
        points = []
        for theta, phi in itertools.product(thetas, phis):
            x = self._a * np.sin(theta) * np.cos(phi)
            y = self._b * np.sin(theta) * np.sin(phi)
            z = self._b * np.cos(theta)

            points.append(self._transform_standard_to_world(np.array([x, y, z])))
        return points
