from swiftsimio.accelerated import concatenate_ranges
from numpy import array, array_equal


def test_concatenate():
    ranges = array([[1, 10], [11, 15], [17, 20], [20, 25]])
    correct = array([[1, 15], [17, 25]])

    concatenated = concatenate_ranges(ranges)

    assert array_equal(concatenated, correct)
    return
