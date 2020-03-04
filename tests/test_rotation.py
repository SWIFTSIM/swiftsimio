"""
Tests the rotation matrix calculations.
"""

from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
import numpy as np


def test_basic_rotation_with_vector():
    """
    Tests to see if the expected rotation matrix rotates given vectors.
    """

    # Note original direction is set to be 0.0, 0.0, 1.0
    test_vectors = 2.0 * (np.random.rand(300).reshape((100, 3)) - 0.5)

    for test_vector in test_vectors:
        for index, direction in enumerate(["x", "y", "z"]):
            rotation_matrix = rotation_matrix_from_vector(test_vector, direction)

            rotated_vector = np.matmul(
                rotation_matrix, test_vector / np.linalg.norm(test_vector)
            )

            expected_vector = np.array([0, 0, 0], dtype=np.float64)
            expected_vector[index] = 1.0

            assert np.isclose(expected_vector, rotated_vector).all()
