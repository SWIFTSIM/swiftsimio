"""
Functions that can be accelerated by numba. Numba does not use classes, unfortunately.
"""

import numpy as np

from h5py._hl.dataset import Dataset

from typing import Tuple
from numba.typed import List
from itertools import chain

try:
    from numba import jit, prange
    from numba.core.config import NUMBA_NUM_THREADS as NUM_THREADS
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

    output_shape: Tuple
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


################################ ALEXEI: playing around with better read_ranges_from_file implementation #################################


def concatenate_ranges(ranges) -> np.ndarray:
    concatenated = []
    concatenated.append(ranges[0])

    for i in range(1, len(ranges)):
        lower = ranges[i][0]
        upper = ranges[i][1]
        if lower <= concatenated[-1][1] + 1:
            concatenated[-1][1] = upper
        else:
            concatenated.append(ranges[i])

    return np.asarray(concatenated)


@jit(nopython=True, fastmath=True)
def get_chunk_ranges(ranges, chunk_size, array_length) -> np.ndarray:
    """
    Return indices indicating which hdf5 chunk each range from `ranges` belongs to

    Parameters
    ----------
    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`)
    chunk_size : int
        size of the hdf5 dataset chunks
    array_length: int
        size of the dataset

    Returns
    -------
    np.ndarray
        two dimensional array of bounds for the chunks that contain each range from
        `ranges`

    """
    chunk_ranges = []
    n_ranges = len(ranges)
    for bounds in ranges:
        lower = (bounds[0] // chunk_size) * chunk_size
        upper = min(-((-bounds[1]) // chunk_size) * chunk_size, array_length)

        # Before appending new chunk range we need to check
        # that it doesn't already exist or overlap with an
        # existing one. The only way overlap can happen is
        # if the current lower index is less than or equal
        # to the previous upper one. In that case simply
        # update the previous upper to cover current chunk
        if len(chunk_ranges) > 0:
            if lower > chunk_ranges[-1][1]:
                chunk_ranges.append([lower, upper])
            elif lower <= chunk_ranges[-1][1]:
                chunk_ranges[-1][1] = upper
        # If chunk_ranges is empty, don't do any checks
        else:
            chunk_ranges.append([lower, upper])

    return np.asarray(chunk_ranges)


@jit(nopython=True, fastmath=True)
def expand_ranges(ranges: np.ndarray) -> np.array:
    """
    Return an array of indices that are within the specified ranges

    Parameters
    ----------
    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`)

    Returns
    -------
    np.array
        1D array of indices that fall within each range specified in `ranges`
        
    """
    length = np.asarray([bounds[1] - bounds[0] for bounds in ranges]).sum()

    output = np.zeros(length, dtype=np.int64)
    i = 0
    for bounds in ranges:
        lower = bounds[0]
        upper = bounds[1]
        bound_length = upper - lower
        output[i : i + bound_length] = np.arange(lower, upper, dtype=np.int64)
        i += bound_length

    return output


@jit(nopython=True, fastmath=True)
def extract_ranges_from_chunks(
    array: np.ndarray, chunks: np.ndarray, ranges: np.ndarray
) -> np.ndarray:
    """
    Returns elements from array that are located within specified ranges
    
    `array` is a portion of the dataset being read consisting of all the chunks
    that contain the ranges specified in `ranges`. The `chunks` array contains
    the indices of the upper and lower bounds of these chunks. To find the 
    elements of the dataset that lie within the specified ranges we first create
    an array indexing which chunk each range belongs to. From this information 
    we create an array of adjusted ranges that takes into account that the array
    is not the whole dataset. We then return the values in `array` that are 
    within the adjusted ranges.
    
    Parameters
    ----------
    array : np.ndarray
        array containing data read in from snapshot
    chunks : np.ndarray
        two dimensional array of bounds for the chunks that contain each range from
        `ranges`
    ranges: np.ndarray
        Array of ranges (see :func:`ranges_from_array`)

    Returns
    -------
    np.ndarray
        subset of `array` whose elements are within each range in `ranges`
    
    """
    # Find out which of the chunks in the chunks array each range in ranges belongs to
    n_ranges = len(ranges)
    chunk_array_index = np.zeros(len(ranges), dtype=np.int32)
    chunk_index = 0
    i = 0
    while i < n_ranges:
        if (
            chunks[chunk_index][0] <= ranges[i][0]
            and chunks[chunk_index][1] >= ranges[i][1]
        ):
            chunk_array_index[i] = chunk_index
            i += 1
        else:
            chunk_index += 2

    # Need to get the locations of the range boundaries with
    # respect to the indexing in the array of chunked data
    # (as opposed to the whole dataset)
    adjusted_ranges = ranges
    running_sum = 0
    for i in range(n_ranges):
        offset = chunks[chunk_array_index[i]][0] - running_sum
        adjusted_ranges[i][0] = ranges[i][0] - offset
        adjusted_ranges[i][1] = ranges[i][1] - offset
        if i < n_ranges:
            if chunk_array_index[i + 1] > chunk_array_index[i]:
                running_sum += (
                    chunks[chunk_array_index[i]][1] - chunks[chunk_array_index[i]][0]
                )

    return array[expand_ranges(adjusted_ranges)]


def new_read_ranges_from_file(
    handle: Dataset,
    ranges: np.ndarray,
    output_shape: Tuple,
    output_type: type = np.float64,
    columns: np.lib.index_tricks.IndexExpression = np.s_[:],
    chunk_size=10000,
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

    output_shape: Tuple
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

    # Get chunk size
    if handle.chunks is not None:
        chunk_size = handle.chunks[0]

        # Make array of chunk ranges
        chunk_ranges = get_chunk_ranges(ranges, chunk_size, handle.shape[0])
        chunk_range_size = int(
            np.sum([chunk_range[1] - chunk_range[0] for chunk_range in chunk_ranges])
        )
        if isinstance(output_shape, tuple):
            output_shape = (chunk_range_size, output_shape[1])
        else:
            output_shape = chunk_range_size
    else:
        chunk_ranges = ranges

    output = np.empty(output_shape, dtype=output_type)
    already_read = 0
    handle_multidim = handle.ndim > 1

    for bounds in chunk_ranges:
        read_start = bounds[0]
        read_end = bounds[1]
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

    if handle.chunks is not None:
        return extract_ranges_from_chunks(output, chunk_ranges, ranges)
    else:
        return output
