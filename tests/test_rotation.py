"""
Tests the rotation matrix calculations.
"""

from swiftsimio.visualisation.rotation import rotation_matrix_from_vector, rotation_matrix
import numpy as np

def test_rotation_vs_vector():
    """
    Computes a known case of vector rotation using both methods and
    compares.
    """

    vector = rotation_matrix_from_vector(
        np.array([-1.0, 0.0, 0.0])
    )

    angle = rotation_matrix(
       pitch=0.0, yaw=np.pi/2, roll=0.0
    )

    assert np.isclose(vector == angle).all()
