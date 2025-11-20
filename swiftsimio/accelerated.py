"""
Define functions that can be accelerated by numba.

Numba does not use classes, unfortunately.
"""

import numpy as np

from h5py._hl.dataset import Dataset

from .optional_packages import jit, prange, NUM_THREADS
from .file_utils import is_hdfstream_dataset

__all__ = [
    "jit",
    "prange",
    "NUM_THREADS",
    "ranges_from_array",
    "read_ranges_from_file_unchunked",
    "index_dataset",
    "concatenate_ranges",
    "get_chunk_ranges",
    "expand_ranges",
    "extract_ranges_from_chunks",
    "read_ranges_from_file_chunked",
    "read_ranges_from_file",
    "list_of_strings_to_arrays",
]


@jit(nopython=True)
def ranges_from_array(array: np.array) -> np.ndarray:
    """
    Find contiguous ranges of IDs in sorted list of IDs.

    Parameters
    ----------
    array : np.array of int
        Sorted list of IDs.

    Returns
    -------
    np.ndarray
        List of length two arrays corresponding to contiguous
        ranges of IDs (inclusive) in the input array.

    Examples
    --------
    The array:

    .. code-block::

        [0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13]

    would return:

    .. code-block::

        [[0, 4], [5, 8], [9, 10], [11, 14]]
    """
    output = []

    if len(array) == 0:
        # the "empty" mask, from 0 to 0 gets 0 elements
        return np.array([[0, 0]])
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
    output_shape: tuple,
    output_type: type = np.float64,
    columns: slice = np.s_[:],
) -> np.array:
    """
    Read only a selection of index ranges from a dataset that is not chunked.

    Takes a hdf5 dataset, and the set of ranges from
    ranges_from_array, and reads only those ranges from the file.

    Unfortunately this functionality is not built into HDF5.

    Parameters
    ----------
    handle : Dataset
        HDF5 dataset to slice data from.

    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`).

    output_shape : tuple
        Resultant shape of output.

    output_type : type, optional
        ``numpy`` type of output elements. If not supplied, we assume ``np.float64``.

    columns : slice, optional
        Selector for columns if using a multi-dimensional array. If the array is only
        a single dimension this is not used.

    Returns
    -------
    np.ndarray
        Result from reading only the relevant values from ``handle``.
    """
    output = np.empty(output_shape, dtype=output_type)
    already_read = 0
    handle_multidim = handle.ndim > 1

    for read_start, read_end in ranges:
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
        output = output.byteswap(inplace=True).newbyteorder()

        if not output.dtype.isnative:
            raise RuntimeError(
                "Unable to find a native type that is a match to read data."
            )

    return output


def index_dataset(handle: Dataset, mask_array: np.array) -> np.array:
    """
    Index the dataset using the mask array.

    This is not currently a feature of h5py. (March 2019)

    Parameters
    ----------
    handle : Dataset
        Data to be indexed.

    mask_array : np.array
        Mask used to index data.

    Returns
    -------
    np.ndarray
        Subset of the data specified by the mask.
    """
    output_type = handle[0].dtype
    output_size = mask_array.size

    ranges = ranges_from_array(mask_array)

    return read_ranges_from_file(handle, ranges, output_size, output_type)


@jit(nopython=True, fastmath=True)
def concatenate_ranges(ranges: np.ndarray) -> np.ndarray:
    """
    Merge consecutive ranges if there is no gap between them.

    Parameters
    ----------
    ranges : np.ndarray
        Array of ranges (see :func:`~swiftsimio.accelerated.ranges_from_array`).

    Returns
    -------
    np.ndarray
        Two dimensional array of ranges.

    Examples
    --------
    .. code-block:: python

        >>> concatenate_ranges([[1,6],[6,10],[12,16]])
        np.ndarray([[1,10],[12,15]])
    """
    concatenated = [list(ranges[0])]

    for i in range(1, len(ranges)):
        lower = ranges[i][0]
        upper = ranges[i][1]
        if lower <= concatenated[-1][1]:
            concatenated[-1][1] = upper
        else:
            concatenated.append(list(ranges[i]))

    return np.array(concatenated)


@jit(nopython=True, fastmath=True)
def get_chunk_ranges(
    ranges: np.ndarray, chunk_size: np.ndarray, array_length: int
) -> np.ndarray:
    """
    Return indices indicating which hdf5 chunk each range from ``ranges`` belongs to.

    Parameters
    ----------
    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`).

    chunk_size : int
        Size of the hdf5 dataset chunks.

    array_length : int
        Size of the dataset.

    Returns
    -------
    np.ndarray
        Two dimensional array of bounds for the chunks that contain each range from
        ``ranges``.
    """
    chunk_ranges = []
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
    Return an array of indices that are within the specified ranges.

    Parameters
    ----------
    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`).

    Returns
    -------
    np.ndarray
        1D array of indices that fall within each range specified in `ranges`.
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
    Return elements from array that are located within specified ranges.

    ``array`` is a portion of the dataset being read consisting of all the chunks
    that contain the ranges specified in ``ranges``. The ``chunks`` array contains
    the indices of the upper and lower bounds of these chunks. To find the
    elements of the dataset that lie within the specified ranges we first create
    an array indexing which chunk each range belongs to. From this information
    we create an array of adjusted ranges that takes into account that the array
    is not the whole dataset. We then return the values in `array` that are
    within the adjusted ranges.

    Parameters
    ----------
    array : np.ndarray
        Array containing data read in from snapshot.

    chunks : np.ndarray
        Two dimensional array of bounds for the chunks that contain each range from
        ``ranges``.

    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`).

    Returns
    -------
    np.ndarray
        Subset of ``array`` whose elements are within each range in ``ranges``.
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
    output_shape: tuple,
    output_type: type = np.float64,
    columns: slice = np.s_[:],
) -> np.array:
    """
    Read only a selection of index ranges from a dataset that is chunked.

    Takes a hdf5 dataset, and the set of ranges from
    ranges_from_array, and reads only those ranges from the file.

    Unfortunately this functionality is not built into HDF5.

    Parameters
    ----------
    handle : Dataset
        HDF5 dataset to slice data from.

    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`).

    output_shape : tuple
        Resultant shape of output.

    output_type : type, optional
        :mod:`numpy` type of output elements. If not supplied, we assume ``np.float64``.

    columns : slice, optional
        Selector for columns if using a multi-dimensional array. If the array is only
        a single dimension this is not used.

    Returns
    -------
    np.ndarray
        Result from reading only the relevant values from ``handle``.
    """
    chunk_size = handle.chunks[0]

    # Make array of chunk ranges
    chunk_ranges = get_chunk_ranges(ranges, chunk_size, handle.shape[0])
    chunk_range_size = np.diff(chunk_ranges).sum()

    try:
        output_shape = (chunk_range_size, output_shape[1])
    except (TypeError, IndexError):
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
        output = output.byteswap(inplace=True).newbyteorder()

        if not output.dtype.isnative:
            raise RuntimeError(
                "Unable to find a native type that is a match to read data."
            )

    if handle.chunks is not None:
        return extract_ranges_from_chunks(output, chunk_ranges, ranges)
    else:
        return output


def read_ranges_from_hdfstream(
    handle: Dataset,
    ranges: np.ndarray,
    output_shape: tuple,
    output_type: type = np.float64,
    columns: slice = np.s_[:],
) -> np.array:
    """
    Request the specified ranges from the hdfstream server.

    Takes a hdfstream remote dataset, and the set of ranges from
    ranges_from_array, and sends a http request for those ranges.

    Parameters
    ----------
    handle : Dataset
        HDF5 dataset to slice data from.

    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`).

    output_shape : Tuple
        Resultant shape of output.

    output_type : type, optional
        ``numpy`` type of output elements. If not supplied, we assume ``np.float64``.

    columns : slice, optional
        Selector for columns if using a multi-dimensional array. If the array is only
        a single dimension this is not used.

    Returns
    -------
    np.ndarray
        Result from reading only the relevant values from ``handle``.
    """
    # Merge any adjacent ranges
    ranges = concatenate_ranges(ranges)

    # If the ranges are not sorted, get the ordering by start index
    if np.any(ranges[1:,0] < ranges[:-1,1]):
        order = np.argsort(ranges[:,0])
        need_reorder = True
    else:
        order = np.arange(ranges.shape[0], dtype=int)
        need_reorder = False

    # Construct an ordered list of slices to request from the server
    slices = []
    for read_start, read_end in ranges[order]:
        if handle.ndim > 1:
            this_slice = np.s_[read_start:read_end, columns]
        else:
            this_slice = np.s_[read_start:read_end]
        # Skip any zero length slices
        if read_end > read_start:
            slices.append(this_slice)

    # Request the slice data as a single ndarray. Here we read into an existing
    # buffer so that we should get an exception if the data type or shape is
    # not what we expected.
    output = np.empty(output_shape, dtype=output_type)
    if len(slices) > 0:
        handle.request_slices(slices, dest=output)

    # Put the result into the same order as the ranges array
    if need_reorder:
        # Compute the offset into the output array for each slice.
        ranges_read = np.empty_like(ranges)
        offset = 0
        for i in order:
            n = ranges[i,1] - ranges[i,0]
            ranges_read[i,0] = offset
            ranges_read[i,1] = offset + n
            offset += n
        # Copy the slices to a new array in the input slice order
        output_sorted = np.empty_like(output)
        offset = 0
        for start, stop in ranges_read:
            n = stop - start
            output_sorted[offset:offset+n,...] = output[start:stop,...]
            offset += n
        output = output_sorted

    if not output.dtype.isnative:
        # The data type we have read in is the opposite endian-ness to the
        # machine we're on. Convert it here, to save pain down the line.
        output = output.byteswap(inplace=True).newbyteorder()

        if not output.dtype.isnative:
            raise RuntimeError(
                "Unable to find a native type that is a match to read data."
            )

    return output


def read_ranges_from_file(
    handle: Dataset,
    ranges: np.ndarray,
    output_shape: tuple,
    output_type: type = np.float64,
    columns: slice = np.s_[:],
) -> np.array:
    """
    Correctly select which version of ``read_ranges_from_file`` should be used.

    Parameters
    ----------
    handle : Dataset
        HDF5 dataset to slice data from.

    ranges : np.ndarray
        Array of ranges (see :func:`ranges_from_array`).

    output_shape : tuple
        Resultant shape of output.

    output_type : type, optional
        ``numpy`` type of output elements. If not supplied, we assume ``np.float64``.

    columns : slice, optional
        Selector for columns if using a multi-dimensional array. If the array is only
        a single dimension this is not used.

    Returns
    -------
    np.ndarray
        Result from reading only the relevant values from ``handle``.

    See Also
    --------
    read_ranges_from_file_chunked
        Reads data ranges for chunked hdf5 file.

    read_ranges_from_file_unchunked
        Reads data ranges for unchunked hdf5 file.
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
    return (read_ranges_from_hdfstream if is_hdfstream_dataset(handle) else read_ranges)(handle, ranges, output_shape, output_type, columns)


def list_of_strings_to_arrays(lines: list[str]) -> np.array:
    """
    Convert a list of space-delimited values to arrays.

    Parameters
    ----------
    lines : list[str]
        List of strings containing numbers separated by a set of spaces.

    Returns
    -------
    list[np.array]
        List of numpy arrays, one per column.

    Notes
    -----
    Currently not suitable for :mod:`numba` acceleration due to mixed datatype usage.
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
