"""
Tests for the accelerated functions.
"""

from swiftsimio.accelerated import (
    ranges_from_array,
    read_ranges_from_file,
    index_dataset,
    list_of_strings_to_arrays,
)

import numpy as np
import h5py

from helper import create_in_memory_hdf5


def test_ranges_from_array():
    """
    Tests ranges from array using the example given.
    """

    my_array = np.array([0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13], dtype=int)

    out = np.array([[0, 4], [5, 8], [9, 10], [11, 14]])

    assert (ranges_from_array(my_array) == out).all()

    return


def test_ranges_from_array_non_contiguous():
    """
    Tests the ranges from array funciton with no contiguous input.
    """

    my_array = np.array([77, 34483, 234582, 123412341324], dtype=int)

    out = np.array(
        [[77, 78], [34483, 34484], [234582, 234583], [123412341324, 123412341325]]
    )

    assert (ranges_from_array(my_array) == out).all()


def test_read_ranges_from_file():
    """
    Tests the reading of ranges from file using a numpy array as a stand in for
    the dataset.
    """

    # In memory hdf5 file
    file_handle = create_in_memory_hdf5()
    handle = file_handle.create_dataset("test", data=np.arange(1000))
    ranges = np.array([[77, 79], [81, 81], [88, 98], [204, 204]])
    output_size = 2 + 10
    output_type = type(handle[0])

    expected = np.array([77, 78, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97])
    out = read_ranges_from_file(handle, ranges, output_size, output_type)

    assert (out == expected).all()

    file_handle.close()


def test_index_dataset():
    """
    Tests the index_dataset function using a numpy array to approximate
    a dataset.
    """

    file = create_in_memory_hdf5()
    data = file.create_dataset("test", data=np.arange(1000))
    mask = np.unique(np.random.randint(0, 1000, 100))

    true = data[list(mask)]

    assert (index_dataset(data, mask) == true).all()

    file.close()


def test_index_dataset_h5py():
    """
    Tests the index_dataset function on a real HDF5 dataset.
    """

    file = create_in_memory_hdf5()

    data = np.arange(100000)
    mask = np.unique(np.random.randint(0, 100000, 10000))

    dataset = file.create_dataset("Test", data=data)

    assert (index_dataset(dataset, mask) == data[mask]).all()


def test_list_of_strings_to_arrays():
    """
    Tests list_of_strings_to_arrays.
    """

    lines = [
        "    0     0.0000    1.0e-3    14.0",
        "    7     3.0000    1.0e-3    14.0",
    ]

    expected_output = [
        np.array([0, 7], dtype=np.int64),
        np.array([0, 3], dtype=np.float64),
        np.array([1e-3, 1e-3], dtype=np.float64),
        np.array([14, 14], dtype=np.float64),
    ]

    output = list_of_strings_to_arrays(lines)

    for expected, real in zip(expected_output, output):
        assert expected.dtype == real.dtype
        assert (expected == real).all()

    return
