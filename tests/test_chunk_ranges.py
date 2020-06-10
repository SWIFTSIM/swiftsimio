from swiftsimio.accelerated import get_chunk_ranges
from numpy import array, array_equal


def test_get_chunk_ranges():
    ranges = array([[0, 2], [2, 4], [11, 17], [21, 23], [24, 25]])
    chunk_size = 5
    length = 25

    correct_chunk_ranges = array([[0, 5], [10, 25]])
    chunk_ranges = get_chunk_ranges(ranges, chunk_size, length)

    assert array_equal(chunk_ranges, correct_chunk_ranges)

    return
