"""
Define functions that can be accelerated by numba.

Numba does not use classes, unfortunately.
"""

import numpy as np
import h5py
from h5py._hl.dataset import Dataset

from .optional_packages import jit, prange, NUM_THREADS
from ._ordered_slices import OrderedSlices

__all__ = [
    "jit",
    "prange",
    "NUM_THREADS",
    "ranges_from_array",
    "read_ranges_from_file_unchunked",
    "index_dataset",
    "get_chunk_ranges",
    "expand_ranges",
    "extract_ranges_from_chunks",
    "read_ranges_from_file_chunked",
    "read_ranges_from_file",
    "read_ranges_from_hdfstream",
    "list_of_strings_to_arrays",
    "to_native_byteorder_inplace",
]


def to_native_byteorder_inplace(arr: np.ndarray) -> None:
    """
    Ensure that arr is native endian without making a copy.

    Parameters
    ----------
    arr : np.ndarray
        Array to convert to native endian.
    """
    if arr.dtype.isnative:
        return
    if arr.dtype.names is None:  # standard check for not a structured or recarray
        arr.byteswap(inplace=True)
    else:
        for name in arr.dtype.names:
            if not arr[name].dtype.isnative:
                arr[name].byteswap(inplace=True)
    arr.dtype = arr.dtype.newbyteorder("native")


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
        :class:`numpy.dtype` of output elements. If not supplied, we assume :py:attr:`numpy.float64`.

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

    # Ensure the result is a native endian type
    to_native_byteorder_inplace(output)

    return output


def read_ranges_from_file_low_level(
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

    This version uses the h5py low level API.

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

    # This will only work if slices do not overlap
    order = np.argsort(ranges[:,0])
    sorted_starts = ranges[order,0]
    sorted_stops  = ranges[order,1]
    if np.any(sorted_stops[:-1] > sorted_starts[1:]):
        raise RuntimeError("slices must not overlap")

    # Get dataset handle
    dataset_id = handle.id

    # Get file dataspace handle
    file_space_id = dataset_id.get_space()
    file_shape = file_space_id.get_simple_extent_dims()

    # Determine range of elements to read in the second dimension (if any)
    if len(handle.shape) == 1:
        column_start = ()
        column_count = ()
    elif len(handle.shape) == 2:
        if isinstance(columns, slice):
            start, stop, step = columns.indices(handle.shape[1])
            if step != 1:
                raise RuntimeError("Can only handle column slices with step=1")
            column_start = (start,)
            column_count = (stop-start,)
        elif isinstance(columns, int):
            column_start = (columns,)
            column_count = (1,)
        else:
            raise RuntimeError("columns parameter must be slice or integer")
    else:
        raise RuntimeError("Can only handle 1 or 2 dimensional datasets")

    # Select the slices to read
    nr_in_first_dim = 0
    file_space_id.select_none()
    for start, stop in ranges:
        count = stop - start
        if count > 0:
            # Select this slice
            slice_start = (start,)+column_start
            slice_count = (count,)+column_count
            file_space_id.select_hyperslab(slice_start, slice_count, op=h5py.h5s.SELECT_OR)
            nr_in_first_dim += count

    # Allocate the output array
    result_shape = (nr_in_first_dim,)+column_count
    result = np.ndarray(result_shape, dtype=output_type)

    # Output array must have the expected number of elements
    nr_selected = file_space_id.get_select_npoints()
    if nr_selected != result.size:
        raise RuntimeError("Output buffer is not the right size for the selected slices!")

    # If we selected any elements, read the data
    if nr_in_first_dim > 0:
        mem_space_id = h5py.h5s.create_simple(result_shape)
        dataset_id.read(mem_space_id, file_space_id, result)
        mem_space_id.close()
    file_space_id.close()

    # Reshape: if columns was an integer we need to remove a dimension
    result = result.reshape(output_shape)

    # If the slices were not sorted by start index, we'll need to reorder the data
    if np.any(ranges[1:,0] <= ranges[:-1,0]):
        # Compute the offset into the result array for each slice.
        # HDF5 reads the slices in order of starting index.
        ranges_read = np.empty_like(ranges)
        offset = 0
        for i in np.argsort(ranges[:,0]):
            n = ranges[i,1] - ranges[i,0]
            ranges_read[i,0] = offset
            ranges_read[i,1] = offset + n
            offset += n
        # Copy the slices to a new array in the input slice order
        result_sorted = np.empty_like(result)
        offset = 0
        for start, stop in ranges_read:
            n = stop - start
            result_sorted[offset:offset+n,...] = result[start:stop,...]
            offset += n
        result = result_sorted

    if not result.dtype.isnative:
        # The data type we have read in is the opposite endian-ness to the
        # machine we're on. Convert it here, to save pain down the line.
        result = result.byteswap(inplace=True).newbyteorder()

        if not result.dtype.isnative:
            raise RuntimeError(
                "Unable to find a native type that is a match to read data."
            )

    return result


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
        :class:`numpy.dtype` of output elements. If not supplied, we assume :py:attr:`numpy.float64`.

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

    if handle.chunks is not None:
        output = extract_ranges_from_chunks(output, chunk_ranges, ranges)

    # Ensure the result is a native endian type
    to_native_byteorder_inplace(output)

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
        :class:`numpy.dtype` of output elements. If not supplied, we assume :py:attr:`numpy.float64`.

    columns : slice, optional
        Selector for columns if using a multi-dimensional array. If the array is only
        a single dimension this is not used.

    Returns
    -------
    np.ndarray
        Result from reading only the relevant values from ``handle``.
    """
    # Get a sorted list of slices to request from the server
    ordered_slices = OrderedSlices(ranges, handle.ndim, columns)

    # Request the slice data as a single ndarray. Here we read into an existing
    # buffer so that we should get an exception if the data type or shape is
    # not what we expected.
    output = np.empty(output_shape, dtype=output_type)
    if len(ordered_slices.slices) > 0:
        handle.request_slices(ordered_slices.slices, dest=output)

    # Put the result into the same order as the input ranges array, if it isn't already
    output = ordered_slices.reorder_result(output)

    # Ensure the result is a native endian type
    to_native_byteorder_inplace(output)

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
        :class:`numpy.dtype` of output elements. If not supplied, we assume :py:attr:`numpy.float64`.

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
    return (
        read_ranges_from_hdfstream if hasattr(handle, "request_slices") else read_ranges_from_file_low_level
    )(handle, ranges, output_shape, output_type, columns)


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
