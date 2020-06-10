from swiftsimio.accelerated import extract_ranges_from_chunks, get_chunk_ranges
import numpy as np


def test_expand_ranges():
    length = 15
    chunk_size = 5
    data = np.arange(length, dtype=np.int32)
    ranges = np.array([[0, 2], [2, 4], [7, 8], [10, 15]])
    chunk_ranges = get_chunk_ranges(ranges, chunk_size, length)

    correct_extraction = np.array([0, 1, 2, 3, 7, 10, 11, 12, 13, 14])
    extracted = extract_ranges_from_chunks(data, chunk_ranges, ranges)

    assert np.array_equal(extracted, correct_extraction)

    return
