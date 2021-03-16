"""
Functions that can be accelerated by numba. Numba does not use classes, unfortunately.
"""

import numpy as np

from h5py._hl.dataset import Dataset

from typing import Tuple, Union, List

try:
    from numba import jit, prange
    from numba.core.config import NUMBA_NUM_THREADS as NUM_THREADS
except (ImportError, ModuleNotFoundError):
    try:
        from numba import jit, prange
        from numba.config import NUMBA_NUM_THREADS as NUM_THREADS
    except (ImportError, ModuleNotFoundError):
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


def read_ranges_from_file_unchunked(
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

    if not output.dtype.isnative:
        # The data type we have read in is the opposite endian-ness to the
        # machine we're on. Convert it here, to save pain down the line.
        output.byteswap().newbyteorder()

        if not output.dtype.isnative:
            raise RuntimeError(
                "Unable to find a native type that is a match to read data."
            )

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


@jit(nopython=True, fastmath=True)
def concatenate_ranges(ranges: np.ndarray) -> np.ndarray:
    """
    Returns an array of ranges with consecutive ranges merged if there is no
    gap between them


    Parameters
    ----------
    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`)

    Returns
    -------
    np.ndarray
        two dimensional array of ranges

    Examples
    --------
    >>> concatenate_ranges([[1,5],[6,10],[12,15]])
    np.ndarray([[1,10],[12,15]])
    """
    concatenated = [list(ranges[0])]

    for i in range(1, len(ranges)):
        lower = ranges[i][0]
        upper = ranges[i][1]
        if lower <= concatenated[-1][1] + 1:
            concatenated[-1][1] = upper
        else:
            concatenated.append(list(ranges[i]))

    return np.array(concatenated)


@jit(nopython=True, fastmath=True)
def get_chunk_ranges(
    ranges: np.ndarray, chunk_size: np.ndarray, array_length: int
) -> np.ndarray:
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

    return np.array(chunk_ranges)


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
        for j in range(bound_length):
            output[i + j] = lower + j
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
    n_chunks = len(chunks)
    chunk_array_index = np.zeros(n_ranges, dtype=np.int32)
    chunk_index = 0
    ranges_index = 0
    while ranges_index < n_ranges and chunk_index < n_chunks:
        if (
            chunks[chunk_index][0] <= ranges[ranges_index][0]
            and chunks[chunk_index][1] >= ranges[ranges_index][1]
        ):
            chunk_array_index[ranges_index] = chunk_index
            ranges_index += 1
        else:
            chunk_index += 1

    # Need to get the locations of the range boundaries with
    # respect to the indexing in the array of chunked data
    # (as opposed to the whole dataset)
    adjusted_ranges = np.copy(ranges)
    running_sum = 0
    for i in range(n_ranges - 1):
        this_chunk = chunks[chunk_array_index[i]]
        offset = this_chunk[0] - running_sum
        adjusted_ranges[i] = ranges[i] - offset
        if chunk_array_index[i + 1] > chunk_array_index[i]:
            running_sum += this_chunk[1] - this_chunk[0]
    # Take care of the last element
    offset = chunks[chunk_array_index[-1]][0] - running_sum
    adjusted_ranges[-1] = ranges[-1] - offset

    return array[expand_ranges(adjusted_ranges)]


def read_ranges_from_file_chunked(
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

    chunk_size = handle.chunks[0]

    # Make array of chunk ranges
    chunk_ranges = get_chunk_ranges(ranges, chunk_size, handle.shape[0])
    chunk_range_size = np.diff(chunk_ranges).sum()

    try:
        output_shape = (chunk_range_size, output_shape[1])
    except:
        # Output shape is just a number, we have a 1D array.
        output_shape = chunk_range_size

    output = np.empty(output_shape, dtype=output_type)
    already_read = 0
    handle_multidim = handle.ndim > 1

    for bounds in chunk_ranges:
        read_start, read_end = bounds
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

    if not output.dtype.isnative:
        # The data type we have read in is the opposite endian-ness to the
        # machine we're on. Convert it here, to save pain down the line.
        output.byteswap().newbyteorder()

        if not output.dtype.isnative:
            raise RuntimeError(
                "Unable to find a native type that is a match to read data."
            )

    if handle.chunks is not None:
        return extract_ranges_from_chunks(output, chunk_ranges, ranges)
    else:
        return output


def read_ranges_from_file(
    handle: Dataset,
    ranges: np.ndarray,
    output_shape: Tuple,
    output_type: type = np.float64,
    columns: np.lib.index_tricks.IndexExpression = np.s_[:],
) -> np.array:
    """
    Wrapper function to correctly select which version of read_ranges_from_file
    should be used

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

    See Also
    --------
    read_ranges_from_file_chunked: reads data within specified ranges for chunked hdf5
    file read_ranges_from_file_unchunked: reads data within specified ranges for 
    unchunked hdf5 file
    """

    # It was found that the range size for which read_ranges_from_file_chunked was
    # faster than unchunked was approximately 5e5. For ranges larger than this the
    # overheads associated with read_ranges_from_file_chunked caused slightly worse
    # performance than read_ranges_from_file_unchunked
    cross_over_range_size = 5e5

    average_range_size = np.diff(ranges).mean()
    read_ranges = (
        read_ranges_from_file_chunked
        if handle.chunks is not None and average_range_size < cross_over_range_size
        else read_ranges_from_file_unchunked
    )

    return read_ranges(handle, ranges, output_shape, output_type, columns)


def list_of_strings_to_arrays(lines: List[str]) -> Union[np.array]:
    """
    Converts a list of space-delimited values to arrays.

    Parameters
    ----------

    lines: List[str]
        List of strings containing numbers separated by a set of spaces.
    
    
    Returns
    -------

    arrays: List[np.array]
        List of numpy arrays, one per column.


    Notes
    -----

    Currently not suitable for ``numba`` acceleration due to mixed datatype usage.
    """

    # Calculate types and set up arrays.

    arrays = []
    dtypes = []
    number_of_lines = len(lines)

    for item in lines[0].split():
        if "." in item or "e" in item:
            dtype = np.float64
        else:
            dtype = np.int64

        dtypes.append(dtype)

        arrays.append(np.zeros(number_of_lines, dtype=dtype))

    for index, line in enumerate(lines):
        for dtype, (array, value) in zip(dtypes, enumerate(line.split())):
            arrays[array][index] = dtype(value)

    return arrays

