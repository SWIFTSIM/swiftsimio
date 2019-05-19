"""
Contains visualisation functions that are accelerated with numba
if it is available.
"""

from typing import Union
from math import sqrt
from numpy import float32, int32, zeros, array, arange, max, ndarray, ones
from swiftsimio import SWIFTDataset

from swiftsimio.accelerated import jit

kernel_gamma = 1.778002
kernel_constant = 80.0 * 3.14159 / 7.0
zero_int32 = int32(0)


@jit(nopython=True, fastmath=True)
def kernel(r: Union[float, float32], h: Union[float, float32]):
    """
    Kernel implementation for swiftsimio. This is the Wendland-C2
    kernel as shown in Denhen & Aly (2012).

    Give it a radius and a smoothing length, and it returns the
    contribution to the density.
    """
    H = h * kernel_gamma
    ratio = r / H

    kernel = 0.0

    if ratio < 1.0:
        ratio_2 = ratio * ratio
        ratio_3 = ratio_2 * ratio

        if ratio < 0.5:
            kernel += 3.0 * ratio_3 - 3.0 * ratio_2 + 0.5

        else:
            kernel += -1.0 * ratio_3 + 3.0 * ratio_2 - 3.0 * ratio + 1.0

    kernel *= kernel_constant

    return kernel / (H * H)


@jit(nopython=True)
def pair_max(a: int32, b: int32) -> int32:
    if a > b:
        return a
    else:
        return b


@jit(nopython=True)
def pair_min(a: int32, b: int32) -> int32:
    if a < b:
        return a
    else:
        return b


@jit(nopython=True, fastmath=True)
def scatter(x: float32, y: float32, m: float32, h: float32, res: int) -> ndarray:
    """
    Creates a scatter plot of:

    + x: the x-positions of the particles. Must be bounded by [0, 1].
    + y: the y-positions of the particles. Must be bounded by [0, 1].
    + m: the masses (or otherwise weights) of the particles
    + h: the smoothing lengths of the particles
    + res: the number of pixels.
    
    This ignores boundary effects.
    """
    # Output array for our image
    image = zeros((res, res), dtype=float32)
    maximal_array_index = int32(res)

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    float_res = float32(res)
    pixel_width = 1.0 / float_res

    # If the kernel width is smaller than this, we drop to just PIC method
    drop_to_single_cell = pixel_width * 0.5

    # Pre-calculate this constant for use with the above
    inverse_cell_area = res * res

    for x_pos, y_pos, mass, hsml in zip(x, y, m, h):
        # Calculate the cell that this particle
        particle_cell_x = int32(float_res * x_pos)
        particle_cell_y = int32(float_res * y_pos)

        # SWIFT stores hsml as the FWHM.
        kernel_width = kernel_gamma * hsml

        if kernel_width < drop_to_single_cell:
            # Easygame, gg
            image[particle_cell_x, particle_cell_y] += mass * inverse_cell_area
        else:
            # The number of cells that this kernel spans
            cells_spanned = int32(1.0 + kernel_width * float_res)

            # Now we loop over the square of cells that the kernel lives in
            for cell_x in range(
                # Ensure that the lowest x value is 0, otherwise we'll segfault
                pair_max(0, particle_cell_x - cells_spanned),
                # Ensure that the highest x value lies within the array bounds,
                # otherwise we'll segfault (oops).
                min(particle_cell_x + cells_spanned, maximal_array_index),
            ):
                # The distance in x to our new favourite cell -- remember that our x, y
                # are all in a box of [0, 1].
                distance_x = float32(cell_x) * pixel_width - x_pos
                distance_x_2 = distance_x * distance_x
                for cell_y in range(
                    pair_max(0, particle_cell_y - cells_spanned),
                    min(particle_cell_y + cells_spanned, maximal_array_index),
                ):
                    distance_y = float32(cell_y) * pixel_width - y_pos
                    distance_y_2 = distance_y * distance_y

                    r = sqrt(distance_x_2 + distance_y_2)

                    kernel_eval = kernel(r, hsml)

                    image[cell_x, cell_y] += mass * kernel_eval

    return image


def project_gas_pixel_grid(
    data: SWIFTDataset, resolution: int, project: Union[str, None] = "masses"
):
    r"""
    Creates a 2D projection of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return:
        \bar{T} = \sum_j T_j W_{ij}
    ).
    
    Default projection variable is mass. If it is None, then we don't
    weight.

    Creates a resolution x resolution array and returns it, without appropriate
    units.
    """

    number_of_gas_particles = data.gas.particle_ids.size

    if project is None:
        m = ones(number_of_gas_particles, dtype=float32)
    else:
        m = getattr(data.gas, project).value

    box_x, box_y, _ = data.metadata.boxsize

    # Let's just hope that the box is square otherwise we're probably SOL
    x, y, _ = data.gas.coordinates.T
    hsml = data.gas.smoothing_length

    image = scatter(x / box_x, y / box_y, m, hsml / box_x, resolution)

    return image


def project_gas(
    data: SWIFTDataset, resolution: int, project: Union[str, None] = "masses"
):
    r"""
    Creates a 2D projection of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return:
        \bar{T} = \sum_j T_j W_{ij}
    ).
    
    Default projection variable is mass. If it is None, then we don't
    weight.

    Creates a resolution x resolution array and returns it, with appropriate
    units.
    """

    image = project_gas_pixel_grid(data, resolution, project)

    units = 1.0 / data.metadata.boxsize[0] * data.metadata.boxsize[1]

    if project is not None:
        units *= getattr(data.gas, project).units

    return image * units

