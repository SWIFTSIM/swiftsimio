"""Provide a class to convert range arrays to ordered slices."""

import numpy as np


class OrderedSlices:
    """
    Class to create an ordered slice list from a range array.

    The hdfstream server can accept requests for multiple slices, but
    they must be sorted by starting index. We also want to eliminate
    empty slices and merge adjacent slices before making the request
    for efficiency. This class takes an array of ranges (as used in
    accelerated.py) as input and converts it to a list of slices.

    It also provides a method to put the array returned by the server
    back into the order in which the ranges were specified.

    Parameters
    ----------
    ranges : np.ndarray
        The array of ranges to request.
    ndim : int
        The number of dimensions of the dataset.
    columns : slice or int, optional
        Selector for columns if using a multi-dimensional array. If the array is only
        a single dimension this is not used. Defaults to all columns.
    """

    def __init__(
        self, ranges: np.ndarray, ndim: int, columns: slice | int = np.s_[:]
    ) -> None:
        # Drop any zero length ranges
        keep = (ranges[:, 1] - ranges[:, 0]) > 0
        ranges = ranges[keep, :]
        nr_ranges = ranges.shape[0]

        # If the ranges are not sorted, get the ordering by start index
        if np.any(ranges[1:, 0] < ranges[:-1, 1]):
            order = np.argsort(ranges[:, 0])
            ordered_start = ranges[order, 0]
            ordered_stop = ranges[order, 1]
        else:
            order = None
            ordered_start = ranges[:, 0]
            ordered_stop = ranges[:, 1]

        # We can't handle overlapping ranges
        if np.any(ordered_start[1:] < ordered_stop[:-1]):
            raise RuntimeError("Ranges to request from the server must not overlap!")

        # Store the slice lengths before merging adjacent slices
        lengths = ordered_stop - ordered_start

        # Determine starting indexes to keep: every starting index which is NOT
        # equal to the end of the previous range. Always keep the first.
        keep_start = np.ones(nr_ranges, dtype=bool)
        keep_start[1:] = ordered_start[1:] != ordered_stop[:-1]

        # Determine ending indexes to keep: every end index which is NOT equal
        # to the start of the next slice. Always keep the last one.
        keep_stop = np.ones(nr_ranges, dtype=bool)
        keep_stop[:-1] = ordered_stop[:-1] != ordered_start[1:]

        # Compute start and stop index for the merged slices
        ordered_start = ordered_start[keep_start]
        ordered_stop = ordered_stop[keep_stop]

        # Make an ordered list of slices
        if ndim > 1:
            slices = [
                np.s_[start:stop, columns]
                for start, stop in zip(ordered_start, ordered_stop)
            ]
        else:
            slices = [
                np.s_[start:stop] for start, stop in zip(ordered_start, ordered_stop)
            ]

        # Invert the sorting index so we can restore the original order later
        if order is not None:
            inverse_order = np.empty_like(order)
            inverse_order[order] = np.arange(len(order), dtype=int)
        else:
            inverse_order = None

        self.slices = slices
        self._inverse_order = inverse_order
        self._lengths = lengths

    def reorder_result(self, arr: np.ndarray) -> np.ndarray:
        """
        Put the array returned by the server into the order specified by the ranges.

        Parameters
        ----------
        arr : np.ndarray
            The array to reorder.

        Returns
        -------
        np.ndarray
            The reordered array.
        """
        if self._inverse_order is not None:
            # Split the array back into ranges
            ranges_in_returned_order = np.split(arr, np.cumsum(self._lengths)[:-1])
            # Restore the original order of the ranges
            ranges_in_requested_order = [
                ranges_in_returned_order[i] for i in self._inverse_order
            ]
            # Recombine
            arr = np.concatenate(ranges_in_requested_order)
        return arr
