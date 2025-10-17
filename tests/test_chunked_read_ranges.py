import numpy as np
from swiftsimio.accelerated import (
    concatenate_ranges,
    get_chunk_ranges,
    extract_ranges_from_chunks,
    expand_ranges,
)


def test_concatenate():
    ranges = np.array([[1, 10], [11, 15], [17, 20], [20, 25]])
    correct = np.array([[1, 15], [17, 25]])

    concatenated = concatenate_ranges(ranges)

    assert np.array_equal(concatenated, correct)
    return


def test_get_chunk_ranges():
    ranges = np.array([[0, 2], [2, 4], [11, 17], [21, 23], [24, 25]])
    chunk_size = 5
    length = 25

    correct_chunk_ranges = np.array([[0, 5], [10, 25]])
    chunk_ranges = get_chunk_ranges(ranges, chunk_size, length)

    assert np.array_equal(chunk_ranges, correct_chunk_ranges)

    return


def test_expand_chunked_ranges():
    length = 15
    chunk_size = 5
    data = np.arange(length, dtype=np.int32)
    ranges = np.array([[0, 2], [2, 4], [7, 8], [10, 15]])
    chunk_ranges = get_chunk_ranges(ranges, chunk_size, length)

    correct_extraction = np.array([0, 1, 2, 3, 7, 10, 11, 12, 13, 14])
    extracted = extract_ranges_from_chunks(data, chunk_ranges, ranges)

    assert np.array_equal(extracted, correct_extraction)

    return


def test_expand_ranges():
    ranges = np.array([[0, 2], [2, 4], [7, 8], [10, 10]])

    correct_expansion = np.array([0, 1, 2, 3, 7])
    expanded_ranges = expand_ranges(ranges)

    assert np.array_equal(expanded_ranges, correct_expansion)

    return
