"""Rotation matrix calculation routines."""

import numpy as np
import unyt as u

from swiftsimio.optional_packages import ROTATION_AVAILABLE, Rotation


def rotation_matrix_from_vector(vector: np.float64, axis: str = "z") -> np.ndarray:
    """
    Calculate a rotation matrix from a vector.

    The comparison vector is assumed to be along an axis, x, y, or z (by default this is
    z). The resulting rotation matrix gives a rotation matrix to align the co-ordinate
    axes to make the projection be top-down along this axis.

    Parameters
    ----------
    vector : np.ndarray[float64]
        3D vector describing the top-down direction that you wish
        to rotate to. For example, this could be the angular momentum
        vector for a galaxy if you wish to produce a top-down projection.

    axis : str, optional
        String describing the axis to project along. This should be one
        of x, y, or z. Defaults to z.

    Returns
    -------
    np.ndarray[float64]
        Rotation matrix (3x3).
    """
    if not ROTATION_AVAILABLE:
        raise ImportError(
            "The scipy.spatial.transform.Rotation class is required to construct "
            "rotation matrices."
        )

    normed_vector = vector / np.linalg.norm(vector)
    if isinstance(normed_vector, u.unyt_array):
        normed_vector = normed_vector.to_value(u.dimensionless)

    # Directional vector describing the axis we wish to look 'down'
    original_direction = np.zeros(3, dtype=np.float64)
    switch = {"x": 0, "y": 1, "z": 2}

    try:
        original_direction[switch[axis]] = 1.0
    except KeyError:
        raise ValueError(
            f"Parameter axis must be one of x, y, or z. You supplied {axis}."
        )

    rotation, _ = Rotation.align_vectors([original_direction], [normed_vector])

    return rotation.as_matrix()
