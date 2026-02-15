"""Test conversion of arrays to native byte order."""

import pytest
import numpy as np
from swiftsimio.accelerated import to_native_byteorder_inplace


@pytest.mark.parametrize(
    "array",
    [
        np.arange(10, dtype=">i1"),
        np.arange(10, dtype="<i1"),
        np.arange(10, dtype="=i1"),
        np.arange(10, dtype=">i2"),
        np.arange(10, dtype="<i2"),
        np.arange(10, dtype="=i2"),
        np.arange(10, dtype=">i4"),
        np.arange(10, dtype="<i4"),
        np.arange(10, dtype="=i4"),
        np.arange(10, dtype=">i8"),
        np.arange(10, dtype="<i8"),
        np.arange(10, dtype="=i8"),
        np.ones(10, dtype=[("a", ">f4"), ("b", ">f4")]),
        np.ones(10, dtype=[("a", ">f4"), ("b", "<f4")]),
        np.ones(10, dtype=[("a", "<f4"), ("b", ">f4")]),
        np.ones(10, dtype=[("a", "<f4"), ("b", "<f4")]),
        np.ones(10, dtype=[("a", ">f4"), ("b", ">i1")]),
        np.ones(10, dtype=[("a", ">f4"), ("b", "<i1")]),
        np.ones(10, dtype=[("a", "<f4"), ("b", ">i1")]),
        np.ones(10, dtype=[("a", "<f4"), ("b", "<i1")]),
    ],
)
def test_slices_from_ranges(array):
    """Check that we can convert various arrays to native endian."""
    # Store the values so we can check they don't change
    array_backup = array.copy()
    # Convert to native endian if necessary
    to_native_byteorder_inplace(array)
    # Ensure we didn't change any values
    assert np.all(array == array_backup)
    # Array should be native endian now
    assert array.dtype.isnative
