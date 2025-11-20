#!/bin/env python

import h5py
import numpy as np


def validate_slices(starts, counts):
    """
    Sanity check the supplied array of slices

    :param starts: 1D array with starting offset of each slice
    :type  starts: np.ndarray
    :param counts: 1D array with length of each slice
    :type  counts: np.ndarray
    """
    if starts.shape != counts.shape:
        raise RuntimeError("start and count arrays must be the same shape")
    if len(starts.shape) != 1 or len(counts.shape) != 1:
        raise RuntimeError("start and count arrays must be 1D")
    if len(starts) > 1:
        if np.any(starts[1:] < starts[:-1]):
            raise RuntimeError("slices must be in ascending order of start index")
        ends = starts + counts
        if np.any(starts[1:] < ends[:-1]):
            raise RuntimeError("slices must not overlap")
    if np.any(counts < 0):
        raise RuntimeError("slices must have non-negative counts")
    if np.any(starts < 0):
        # We don't support negative indexes
        raise RuntimeError("slices must have non-negative starts")


def merge_slices(starts, counts):
    """
    Given a set of slices where slice i starts at index starts[i] and contains
    counts[i] elements, merge any adjacent slices and return new starts and
    counts arrays.

    :param starts: 1D array with starting offset of each slice
    :type  starts: np.ndarray
    :param counts: 1D array with length of each slice
    :type  counts: np.ndarray

    :return: new (starts, counts) tuple with the merged slices
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    starts = np.asarray(starts, dtype=int)
    counts = np.asarray(counts, dtype=int)

    # First, eliminate any zero length slices
    keep = counts > 0
    starts = starts[keep]
    ends = starts + counts[keep]

    # Determine number of slices
    nr_slices = len(starts)
    if len(ends) != nr_slices:
        raise ValueError("starts and counts arrays must be the same size!")

    # Determine starts to keep: every starting offset which is NOT
    # equal to the end of the previous slice. Always keep the first.
    keep_start = np.ones(nr_slices, dtype=bool)
    keep_start[1:] = (starts[1:] != ends[:-1])

    # Determine ends to keep: every end offset which is NOT equal
    # to the start of the next slice. Always keep the last one.
    keep_end = np.ones(nr_slices, dtype=bool)
    keep_end[:-1] = (ends[:-1] != starts[1:])

    # Discard unwanted elements
    assert len(starts) == len(ends)
    starts = starts[keep_start]
    counts = ends[keep_end] - starts

    return starts, counts


def read_slices(dataset, starts, counts, result=None):
    """
    Read the specified slices from a HDF5 dataset. Uses h5py low level calls
    to read the slices with a single H5Dread(). Datasets can only be sliced
    along the first dimension: we always read all elements in the remaining
    dimensions.

    Slices must be in ascending order of starting index and must not overlap.
    Python/numpy style negative indexes from the end of the dataset are not
    supported.

    :param dataset: HDF5 dataset to read from
    :type  dataset: h5py.Dataset
    :param starts: 1D array with starting offset of each slice
    :type  starts: np.ndarray
    :param counts: 1D array with length of each slice
    :type  counts: np.ndarray
    :param result: array to hold the result
    :type  result: np.ndarray, or None

    :return: a numpy array with the data
    :rtype: numpy.ndarray
    """

    # Sanity check the slices
    starts = np.asarray(starts, dtype=int)
    counts = np.asarray(counts, dtype=int)
    validate_slices(starts, counts)

    # Merged any adjacent slices
    starts, counts = merge_slices(starts, counts)

    # Get dataset handle
    dataset_id = dataset.id

    # Get file dataspace handle
    file_space_id = dataset_id.get_space()
    file_shape = file_space_id.get_simple_extent_dims()

    # Select the slices to read
    nr_in_first_dim = 0
    file_space_id.select_none()
    for start, count in zip(starts, counts):
        if count > 0:
            # Select this slice
            slice_start = tuple([start,]+[0 for fs in file_shape[1:]])
            slice_count = tuple([count,]+[fs for fs in file_shape[1:]])
            file_space_id.select_hyperslab(slice_start, slice_count, op=h5py.h5s.SELECT_OR)
            nr_in_first_dim += count

    # Allocate the output array, if necessary
    result_shape = [nr_in_first_dim,]+list(file_shape[1:])
    result_shape = tuple([int(rs) for rs in result_shape])
    if result is None:
        result = np.ndarray(result_shape, dtype=dataset.dtype)

    # Output array must be C contiguous
    if not result.flags['C_CONTIGUOUS']:
        raise RuntimeError("Can only read into C contiguous arrays!")

    # Output array must have the expected number of elements
    nr_selected = file_space_id.get_select_npoints()
    if nr_selected != result.size:
        raise RuntimeError("Output buffer is not the right size for the selected slices!")

    # The output array must have the expected shape (could be wrong if it was passed in)
    if result.shape != result_shape:
        raise RuntimeError("Output buffer has the wrong shape!")

    # If we selected any elements, read the data
    if nr_in_first_dim > 0:
        mem_space_id = h5py.h5s.create_simple(result_shape)
        dataset_id.read(mem_space_id, file_space_id, result)

    return result


class IndexedDatasetReader:

    def __init__(self, index, sorted_and_unique=False):
        """
        Class for reading specified indexes from HDF5 datasets. Here we assume
        that the requested indexes are likely to include runs of consecutive
        values and so can be efficiently handled using hyperslab reads. The
        array of indexes is converted into (start, count) pairs and datasets
        are read using read_slices().

        The supplied indexes are in the first dimension. We read all data in any
        subsequent dimensions. Indexes must be unique and in ascending order if
        sorted_and_unique is True.

        An instance of this class can be used to read the same elements from
        multiple datasets.

        :param index: 1D array with indexes to read in the first dimension
        :type  index: np.ndarray
        :param sorted_and_unique: set to True if index values are sorted and unique
        :type  sorted_and_unique: bool
        """
        # Get sorted, unique indexes if necessary
        index = np.asarray(index, dtype=int)
        if sorted_and_unique:
            self.unique_index = index
            self.inverse_index = None
        else:
            self.unique_index, self.inverse_index = np.unique(index, return_inverse=True)

        # Every index is a range of length one. Merge any adjacent ranges.
        self.starts, self.counts = merge_slices(self.unique_index, np.ones(len(self.unique_index), dtype=int))

    def read(self, dataset):
        """
        Read the specified indexes from a HDF5 dataset.

        :param dataset: HDF5 dataset to read from
        :type  dataset: h5py.Dataset

        :return: a numpy array with the data read from the dataset
        :rtype: numpy.ndarray
        """
        # Read in the specified ranges
        result = read_slices(dataset, self.starts, self.counts)

        # And put the result into the order in which the indexes were requested
        if self.inverse_index is not None:
            result = result[self.inverse_index,...]
        return result


class SlicedDatasetReader:

    def __init__(self, starts, counts):
        """
        Class for reading specified slices from HDF5 datasets. Datasets are
        read using read_slices(). The supplied slices are in the first
        dimension. We read all data in any subsequent dimensions.

        An instance of this class can be used to read the same elements from
        multiple datasets.

        :param starts: 1D array with starting offset of each slice
        :type  starts: np.ndarray
        :param counts: 1D array with length of each slice
        :type  counts: np.ndarray
        """
        # Merge and store any adjacent ranges
        self.starts, self.counts = merge_slices(starts, counts)

    def read(self, dataset):
        """
        Read the specified indexes from a HDF5 dataset.

        :param dataset: HDF5 dataset to read from
        :type  dataset: h5py.Dataset

        :return: a numpy array with the data read from the dataset
        :rtype: numpy.ndarray
        """
        # Read in the specified ranges
        return read_slices(dataset, self.starts, self.counts)


def match(arr1, arr2, arr2_sorted=False, arr2_index=None):
    """
    For each element in arr1 return the index of the element with the
    same value in arr2, or -1 if there is no element with the same value.
    Setting arr2_sorted=True will save some time if arr2 is already sorted
    into ascending order.

    A precomputed sorting index for arr2 can be supplied using the
    arr2_index parameter. This can save time if the routine is called
    repeatedly with the same arr2 but arr2 is not already sorted.

    It is assumed that each element in arr1 only occurs once in arr2.
    """

    # Check for the case where we're searching an empty arr2 - can't be any matches
    if len(arr2) == 0:
        return -np.ones(len(arr1), dtype=int)

    # Workaround for a numpy bug (<=1.4): ensure arrays are native endian
    # because searchsorted ignores endian flag
    if not(arr1.dtype.isnative):
        arr1_n = np.asarray(arr1, dtype=arr1.dtype.newbyteorder("="))
    else:
        arr1_n = arr1
    if not(arr2.dtype.isnative):
        arr2_n = np.asarray(arr2, dtype=arr2.dtype.newbyteorder("="))
    else:
        arr2_n = arr2

    # Sort arr2 into ascending order if necessary
    tmp1 = arr1_n
    if arr2_sorted:
        tmp2 = arr2_n
        idx = slice(0,len(arr2_n))
    else:
        if arr2_index is None:
            idx = np.argsort(arr2_n)
            tmp2 = arr2_n[idx]
        else:
            # Use supplied sorting index
            idx = arr2_index
            tmp2 = arr2_n[arr2_index]

    # Find where elements of arr1 are in arr2
    ptr  = np.searchsorted(tmp2, tmp1)

    # Make sure all elements in ptr are valid indexes into tmp2
    # (any out of range entries won't match so they'll get set to -1
    # in the next bit)
    ptr[ptr>=len(tmp2)] = 0
    ptr[ptr<0]          = 0

    # Return -1 where no match is found
    ind  = tmp2[ptr] != tmp1
    ptr[ind] = -1

    # Put ptr back into original order
    ind = np.arange(len(arr2_n))[idx]
    ptr = np.where(ptr>= 0, ind[ptr], -1)

    return ptr


