"""
Functions that can be accelerated by numba. Numba does not use classes, unfortunately.
"""

import numpy as np

from h5py._hl.dataset import Dataset

try:
    from numba import jit
except ImportError:
    print(
        "You do not have numba installed. Please consider installing"
        "if you are going to be indexing large arrays"
    )

    def jit(func, *args, **kwargs):
        return func


@jit(nopython=True)
def ranges_from_array(array: np.array) -> np.ndarray:
    """
    Takes in an array of IDs (assumed to be sorted) and returns a list of
    the following structure:

    [
        [i, j]
        [k, l]
        [n, o]
        ...
        [c, c]
        [c, d]
    ]
    
    Which are the ranges that are contiguous (inclusive) in the input array. For example,
    the array

    [0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13]

    would return

    [[0, 3], [5, 7], [9, 9], [11, 13]]

    The input array must have type int.
    """

    output = []

    start = array[0]
    stop = array[0]
    
    for value in array[1:]:
        if value != stop + 1:
            output.append([start, stop])

            start = value
            stop = value
        else:
            stop = value

    output.append([start, stop])

    return np.array(output)


def read_ranges_from_file(handle: Dataset, ranges: np.ndarray, output_size: int, output_type=np.float64) -> np.array:
    """
    Takes a hdf5 dataset, and the set of ranges from
    ranges_from_array, and reads only those ranges from the file.

    Unfortunately this functionality is not built into HDF5 so we have
    to do this ourself in this kind of gross way.
    """

    output = np.empty(output_size, dtype=output_type)
    already_read = 0

    # Cannot do pythonic loop because numba cannot jit that
    for index in np.arange(ranges.shape[0]):
        range = ranges[index]
        # Because we read inclusively
        size_of_range = (range[1] + 1) - range[0]

        output[already_read : size_of_range + already_read] = handle[range[0] : range[1] + 1]

        already_read += size_of_range

    return output


def index_dataset(handle: Dataset, mask_array: np.array) -> np.array:
    """
    Indexes the dataset using the mask array.

    This is not currently a feature of h5py. (March 2019)
    """

    output_type = handle[0].dtype
    output_size = mask_array.size

    ranges = ranges_from_array(mask_array)

    return read_ranges_from_file(handle, ranges, output_size, output_type)

