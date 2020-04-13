"""
Reference evaluation - returns a 2d histogram (i.e. no smoothing).

Uses double precision.
"""


from typing import Union
from math import sqrt, ceil
from numpy import float32, float64, int32, zeros, ndarray

from swiftsimio.accelerated import jit, NUM_THREADS, prange


@jit(nopython=True, fastmath=True)
def scatter(x: float64, y: float64, m: float32, h: float32, res: int) -> ndarray:
    """
    Creates a scatter plot of:
    + x: the x-positions of the particles. Must be bounded by [0, 1].
    + y: the y-positions of the particles. Must be bounded by [0, 1].
    + m: the masses (or otherwise weights) of the particles
    + h: the smoothing lengths of the particles
    + res: the number of pixels.
    This ignores boundary effects.
    Note that explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.
    """
    # Output array for our image
    image = zeros((res, res), dtype=float64)
    maximal_array_index = int32(res)

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    # We need this for combining with the x_pos and y_pos variables.
    float_res = float64(res)
    # Pre-calculate this constant for use with the above
    inverse_cell_area = float_res * float_res

    for x_pos, y_pos, mass in zip(x, y, m):
        # Calculate the cell that this particle; use the 64 bit version of the
        # resolution as this is the same type as the positions
        particle_cell_x = int32(float_res * x_pos)
        particle_cell_y = int32(float_res * y_pos)

        if not (
            particle_cell_x < 0
            or particle_cell_x >= maximal_array_index
            or particle_cell_y < 0
            or particle_cell_y >= maximal_array_index
        ):
            image[particle_cell_x, particle_cell_y] += float64(mass) * inverse_cell_area

    return image


@jit(nopython=True, fastmath=True, parallel=True)
def scatter_parallel(
    x: float64, y: float64, m: float32, h: float32, res: int
) -> ndarray:
    """
    Same as scatter, but executes in parallel! This is actually trivial,
    we just make NUM_THREADS images and add them together at the end.
    """

    number_of_particles = x.size
    core_particles = number_of_particles // NUM_THREADS

    output = zeros((res, res), dtype=float64)

    for thread in prange(NUM_THREADS):
        # Left edge is easy, just start at 0 and go to 'final'
        left_edge = thread * core_particles

        # Right edge is harder in case of left over particles...
        right_edge = thread + 1

        if right_edge == NUM_THREADS:
            right_edge = number_of_particles
        else:
            right_edge *= core_particles

        output += scatter(
            x=x[left_edge:right_edge],
            y=y[left_edge:right_edge],
            m=m[left_edge:right_edge],
            h=h[left_edge:right_edge],
            res=res,
        )

    return output
