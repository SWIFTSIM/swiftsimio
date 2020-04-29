"""
Functions that can be accelerated by numba. Numba does not use classes, unfortunately.
"""

import numpy as np

from h5py._hl.dataset import Dataset

from typing import Tuple

try:
    from numba import jit, prange
    from numba.config import NUMBA_NUM_THREADS as NUM_THREADS
except ImportError:
    print(
        "You do not have numba installed. Please consider installing "
        "if you are going to be doing visualisation or indexing large arrays "
        "(pip install numba)"
    )

    def jit(*args, **kwargs):
        def x(func):
            return func

        return x

    prange = range
    NUM_THREADS = 1


@jit(nopython=True)
def ranges_from_array(array: np.array) -> np.ndarray:
    """
    Finds contiguous ranges of IDs in sorted list of IDs

    Parameters
    ----------
    array : np.array of int
        sorted list of IDs 

    Returns
    -------
    np.ndarray
        list of length two arrays corresponding to contiguous 
        ranges of IDs (inclusive) in the input array
    
    Examples
    --------
    The array

    [0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13]

    would return

    [[0, 4], [5, 8], [9, 10], [11, 14]]

    """

    output = []

    start = array[0]
    stop = array[0]

    for value in array[1:]:
        if value != stop + 1:
            output.append([start, stop + 1])

            start = value
            stop = value
        else:
            stop = value

    output.append([start, stop + 1])

    return np.array(output)


def read_ranges_from_file(
    handle: Dataset,
    ranges: np.ndarray,
    output_shape: Tuple,
    output_type: type = np.float64,
    columns: np.lib.index_tricks.IndexExpression = np.s_[:],
) -> np.array:
    """
    Takes a hdf5 dataset, and the set of ranges from
    ranges_from_array, and reads only those ranges from the file.

    Unfortunately this functionality is not built into HDF5.

    Parameters
    ----------

    handle: Dataset
        HDF5 dataset to slice data from

    ranges: np.ndarray
        Array of ranges (see :func:`ranges_from_array`)

    output_shape: Tuple, optional
        Resultant shape of output. 
    
    output_type: type, optional
        ``numpy`` type of output elements. If not supplied, we assume ``np.float64``.

    columns: np.lib.index_tricks.IndexExpression, optional
        Selector for columns if using a multi-dimensional array. If the array is only
        a single dimension this is not used.

    
    Returns
    -------

    array: np.ndarray
        Result from reading only the relevant values from ``handle``.
    """

    output = np.empty(output_shape, dtype=output_type)
    already_read = 0
    handle_multidim = handle.ndim > 1

    for (read_start, read_end) in ranges:
        if read_end == read_start:
            continue

        # Because we read inclusively
        size_of_range = read_end - read_start

        # Construct selectors so we can use read_direct to prevent creating
        # copies of data from the hdf5 file.
        hdf5_read_sel = (
            np.s_[read_start:read_end, columns]
            if handle_multidim
            else np.s_[read_start:read_end]
        )

        output_dest_sel = np.s_[already_read : size_of_range + already_read]

        handle.read_direct(output, source_sel=hdf5_read_sel, dest_sel=output_dest_sel)

        already_read += size_of_range

    return output


def index_dataset(handle: Dataset, mask_array: np.array) -> np.array:
    """
    Indexes the dataset using the mask array.

    This is not currently a feature of h5py. (March 2019)

    Parameters
    ----------
    handle : Dataset
        data to be indexed
    mask_array : np.array
        mask used to index data

    Returns
    -------
    np.array
        Subset of the data specified by the mask
    """

    output_type = handle[0].dtype
    output_size = mask_array.size

    ranges = ranges_from_array(mask_array)

    return read_ranges_from_file(handle, ranges, output_size, output_type)
