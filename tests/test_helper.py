"""
Tests the helper functions if necessary.
"""

from .helper import create_in_memory_hdf5


def test_create_in_memory_hdf5():
    """
    Tests the creation of an in-memory hdf5 file.
    """
    file = create_in_memory_hdf5()
    file.close()

    return
