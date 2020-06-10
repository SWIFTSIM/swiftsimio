from swiftsimio.accelerated import expand_ranges
from numpy import array, array_equal


def test_expand_ranges():
    ranges = array([[0, 2], [2, 4], [7, 8], [10, 10]])

    correct_expansion = array([0, 1, 2, 3, 7])
    expanded_ranges = expand_ranges(ranges)

    assert array_equal(expanded_ranges, correct_expansion)

    return
