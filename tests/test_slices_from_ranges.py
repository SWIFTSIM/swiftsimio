"""Test conversion of range arrays to lists of slices."""

import pytest
import numpy as np
from swiftsimio._ordered_slices import OrderedSlices


range_cases = [
    # A single range
    {
        "ranges": [
            [0, 10],
        ],
        "slices": [
            np.s_[0:10],
        ],
        "error": None,
    },
    # Two ranges which cannot be merged
    {
        "ranges": [[0, 10], [20, 40]],
        "slices": [np.s_[0:10], np.s_[20:40]],
        "error": None,
    },
    # Two ranges which cannot be merged, in reverse order
    {
        "ranges": [[20, 40], [0, 10]],
        "slices": [np.s_[0:10], np.s_[20:40]],  # output should be sorted
        "error": None,
    },
    # Two ranges which can be merged
    {
        "ranges": [[0, 20], [20, 50]],
        "slices": [np.s_[0:50]],
        "error": None,
    },
    # Two ranges which can be merged, in reverse order
    {
        "ranges": [[20, 50], [0, 20]],
        "slices": [np.s_[0:50]],
        "error": None,
    },
    # Including some zero sized ranges, which should be removed
    {
        "ranges": [
            [5, 10],
            [10, 10],
            [20, 15],
            [15, 10],
            [15, 0],
            [30, 40],
            [40, 40],
            [40, 43],
        ],
        "slices": [np.s_[5:10], np.s_[30:43]],
        "error": None,
    },
    # As above, but not sorted
    {
        "ranges": [
            [15, 10],
            [5, 10],
            [20, 15],
            [40, 40],
            [15, 0],
            [30, 40],
            [40, 43],
            [10, 10],
        ],
        "slices": [np.s_[5:10], np.s_[30:43]],
        "error": None,
    },
    # A case with overlapping ranges, which should fail
    {
        "ranges": [[0, 50], [40, 100]],
        "slices": [
            np.s_[0:100],
        ],
        "error": RuntimeError,
    },
    # Empty overlapping ranges don't matter
    {
        "ranges": [[0, 50], [20, 20]],
        "slices": [
            np.s_[0:50],
        ],
        "error": None,
    },
]

column_cases = [
    # 1D dataset with no column selection
    {
        "ndim": 1,
        "columns": None,
    },
    # 2D dataset with single slice column selection
    {
        "ndim": 2,
        "columns": np.s_[0:3],
    },
    # 2D dataset with integer column selection
    {
        "ndim": 2,
        "columns": 5,
    },
]


def normalize_slice(s: slice) -> tuple:
    """
    Return slice components as int or None (not np.int64!).

    Parameters
    ----------
    s : slice
        The slice object to operate on.

    Returns
    -------
    (start, stop, step)
        The start, stop and step values as int or None.
    """
    return (
        int(s.start) if s.start is not None else None,
        int(s.stop) if s.stop is not None else None,
        int(s.step) if s.step is not None else None,
    )


def slices_equal(s1, s2):
    """
    Return True if slices have the same start, stop and step.

    Parameters
    ----------
    s1 : slice
        The first slice object to operate on.
    s2 : slice
        The second slice object to operate on.

    Returns
    -------
    bool
        True if slices are equivalent.
    """
    return normalize_slice(s1) == normalize_slice(s2)


@pytest.mark.parametrize("range_params", range_cases)
@pytest.mark.parametrize("column_params", column_cases)
def test_slices_from_ranges(range_params, column_params):
    """Convert range array to list of slices and check the result."""
    # Unpack parameters
    ranges = range_params["ranges"]
    expected_slices = range_params["slices"]
    error = range_params["error"]
    ndim = column_params["ndim"]
    columns = column_params["columns"]

    # Sanity check the ranges parameter
    ranges = np.asarray(ranges, dtype=int)
    assert len(ranges.shape) == 2
    assert ranges.shape[1] == 2

    if error is None:
        # Convert ranges to a list of slices
        ordered_slices = OrderedSlices(ranges, ndim, columns)
        assert len(ordered_slices.slices) == len(expected_slices)

        # Check we got the expected slices
        for expected_slice, actual_slice in zip(expected_slices, ordered_slices.slices):
            if ndim == 1:
                assert slices_equal(actual_slice, expected_slice)
            elif ndim == 2:
                # Should have tuple (slice, slice | int).
                # The columns tuple is just passed through.
                assert slices_equal(actual_slice[0], expected_slice)
                assert actual_slice[1] == columns
            else:
                raise RuntimeError("Only implemented for ndim=1 or 2")

        # Create a dataset to try reading
        max_index = np.amax(ranges.flatten())
        n = max_index + 1
        if ndim == 1:
            dataset = np.arange(n, dtype=int)
        elif ndim == 2:
            max_cols = 10
            dataset = np.arange(max_cols * n, dtype=int).reshape((n, max_cols))
        else:
            raise RuntimeError("Only implemented for ndim=1 or 2")

        # Use the input ranges to construct the correct result
        if columns is None:
            expected_data = np.concatenate(
                [dataset[start:stop, ...] for (start, stop) in ranges]
            )
        else:
            assert ndim == 2
            expected_data = np.concatenate(
                [dataset[start:stop, columns] for (start, stop) in ranges]
            )

        # Use the sorted slices to construct a result which may not be ordered correctly
        wrong_order_data = np.concatenate([dataset[s] for s in ordered_slices.slices])

        # Restore the correct ordering. Result should be the same as applying the ranges.
        actual_data = ordered_slices.reorder_result(wrong_order_data)
        assert np.all(actual_data == expected_data)

    else:
        # This case is expected to fail
        with pytest.raises(error):
            ordered_slices = OrderedSlices(ranges, ndim, columns)
