"""
Rotation matrix calculation routines.
"""

from swiftsimio.accelerated import jit
from numpy import float64, array, matrix, cross, identity, dot, matmul
from numpy.linalg import norm, inv
from math import sin, cos, sqrt, acos


def rotation_matrix_from_vector(vector: float64, axis: str = "z") -> array:
    """
    Calculate a rotation matrix from a vector. The comparison vector is
    assumed to be along an axis, x, y, or z (by default this is z). The
    resulting rotation matrix gives a rotation matrix to align the
    co-ordinate axes to make the projection be top-down along this axis.

    Parameters
    ----------

    vector: np.array[float64]
        3D vector describing the top-down direction that you wish
        to rotate to. For example, this could be the angular momentum
        vector for a galaxy if you wish to produce a top-down projection.

    axis: str, optional
        String describing the axis to project along. This should be one
        of x, y, or z. Defaults to z.


    Returns
    -------

    rotation_matrix: np.array[float64]
        Rotation matrix (3x3).
    """

    normed_vector = vector / norm(vector)

    # Directional vector describing the axis we wish to look 'down'
    original_direction = array([0.0, 0.0, 0.0], dtype=float64)
    switch = {"x": 0, "y": 1, "z": 2}

    try:
        original_direction[switch[axis]] = 1.0
    except KeyError:
        raise ValueError(
            f"Parameter axis must be one of x, y, or z. You supplied {axis}."
        )

    cross_product = cross(original_direction, normed_vector)
    mod_cross_product = norm(cross_product)
    cross_product /= mod_cross_product

    if mod_cross_product <= 1e-6:
        # This case only covers when we point the vector
        # in the exact opposite direction (e.g. flip z).
        output = identity(3)
        output[switch[axis], switch[axis]] = -1.0

        return output
    else:
        cos_theta = dot(original_direction, normed_vector)
        sin_theta = sin(acos(cos_theta))

        # Skew symmetric matrix for cross product
        A = array(
            [
                [0.0, -cross_product[2], cross_product[1]],
                [cross_product[2], 0.0, -cross_product[0]],
                [-cross_product[1], cross_product[0], 0.0],
            ]
        )

        return inv(identity(3) + sin_theta * A + (1 - cos_theta) * matmul(A, A))
