"""
Rotation matrix calculation routines.
"""

from swiftsimio.accelerated import jit
from numpy import float64, array, matrix, cross, identity, dot, matmul
from numpy.linalg import norm
from math import sin, cos, sqrt, acos


def rotation_matrix(yaw: float64, pitch: float64, roll:float64) -> array:
    """
    Calculates a 3D rotation matrix for the three given angles.

    Parameters
    ----------

    yaw: float64
        Rotation around top-down central axis (in radians).

    pitch: float64
        Left-right rotation angle (in radians).

    roll: float64
        Front-back rotation angle (in radians).
        
    
    Returns
    -------

    rotation_matrix: np.array[float64]
        Rotation matrix (3x3).


    Examples
    --------

    .. code-block:: python

      yaw = np.pi / 4
      pitch = np.pi
      roll = 0.0

      matrix = rotation_matrix(yaw=yaw, pitch=pitch, roll=roll)

    
    """

    siny = sin(yaw)
    sinp = sin(pitch)
    sinr = sin(roll)

    cosy = cos(yaw)
    cosp = cos(pitch)
    cosr = cos(roll)

    yaw_matrix = array([
        [cosy, -siny, 0.0],
        [siny, cosy, 0.0],
        [0.0, 0.0, 1.0]
    ])

    pitch_matrix = array([
        [cosp, 0.0, sinp],
        [0.0, 1.0, 0.0],
        [-sinp, 0.0, cosp]
    ])

    roll_matrix = array([
        [1.0, 0.0, 0.0],
        [0.0, cosr, -sinr],
        [0.0, sinr, cosr],
    ])

    return matmul(yaw_matrix, matmul(pitch_matrix, roll_matrix))

def rotation_matrix_from_vector(vector: float64) -> array:
    """
    Calculate the rotation matrix from a vector. The starting
    vector is assumed to be pointing towards the viewer, i.e.
    it is [0.0, 0.0, -1.0]. The resulting rotation matrix gives
    a rotation matrix to align the co-ordinate axes to make the
    projection be top-down.

    Parameters
    ----------

    vector: np.array[float64]
        3D vector describing the top-down direction that you wish
        to rotate to. For example, this could be the angular momentum
        vector for a galaxy if you wish to produce a top-down projection.


    Returns
    -------

    rotation_matrix: np.array[float64]
        Rotation matrix (3x3).
    """

    vector = vector / norm(vector)

    original_direction = array([0.0, 0.0, 1.0], dtype=float64)

    cross_product = cross(original_direction, vector)
    mod_cross_product = norm(cross_product)

    if mod_cross_product == 0.0:
        # This case only covers when we point the vector
        # in the exact oposite direction (i.e. flip z).
        return array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
    else:
        theta = acos(dot(original_direction, vector) / norm(vector))

        # Skew symmetric matrix for cross product
        A = array([
            [0.0, -cross_product[2], cross_product[1]],
            [cross_product[2], 0.0, -cross_product[0]],
            [-cross_product[1], cross_product[0], 0.0]
        ])

        return identity(3) + sin(theta) * A + (1.0 - cos(theta)) * matmul(A, A)

