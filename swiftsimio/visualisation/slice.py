"""
Sub-module for slice plots in SWFITSIMio.
"""

from typing import Union
from math import sqrt
from numpy import float64, float32, int32, zeros, array, arange, ndarray, ones
from swiftsimio import SWIFTDataset

from swiftsimio.accelerated import jit

# Taken from Dehnen & Aly 2012
kernel_gamma = 1.936492
kernel_constant = 21.0 / (2.0 * 3.14159)


@jit(nopython=True, fastmath=True)
def kernel(r: Union[float, float32], H: Union[float, float32]):
    """
    Kernel implementation for swiftsimio. This is the Wendland-C2
    kernel as shown in Denhen & Aly (2012).

    Give it a radius and a kernel width (i.e. not a smoothing length, but the
    radius of compact support) and it returns the contribution to the
    density.
    """
    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = 0.0

    if ratio < 1.0:
        ratio_2 = ratio * ratio
        ratio_3 = ratio_2 * ratio

        if ratio < 0.5:
            kernel += 3.0 * ratio_3 - 3.0 * ratio_2 + 0.5

        else:
            kernel += -1.0 * ratio_3 + 3.0 * ratio_2 - 3.0 * ratio + 1.0

        kernel *= kernel_constant * inverse_H * inverse_H

    return kernel


@jit(nopython=True, fastmath=True)
def slice_scatter(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float64,
    res: int,
) -> ndarray:
    """
    Creates a slice-scatter plot of:

    + x: the x-positions of the particles. Must be bounded by [0, 1].
    + y: the y-positions of the particles. Must be bounded by [0, 1].
    + z: the z-positions of the particles. Must be bounded by [0, 1].
    + m: the masses (or otherwise weights) of the particles
    + h: the smoothing lengths of the particles
    + z_slice: the position at which we wish to create the slice
    + res: the number of pixels.
    
    This ignores boundary effects.

    Note that explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.
    """
    # Output array for our image
    image = zeros((res, res), dtype=float32)
    maximal_array_index = int32(res)

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    float_res = float32(res)
    pixel_width = 1.0 / float_res

    # We need this for combining with the x_pos and y_pos variables.
    float_res_64 = float64(res)

    # If the kernel width is smaller than this, we drop to just PIC method
    drop_to_single_cell = pixel_width * 0.5

    # Pre-calculate this constant for use with the above
    inverse_cell_area = res * res

    for x_pos, y_pos, z_pos, mass, hsml in zip(x, y, z, m, h):
        # Calculate the cell that this particle lives above; use 64 bits
        # resolution as this is the same type as the positions
        particle_cell_x = int32(float_res_64 * x_pos)
        particle_cell_y = int32(float_res_64 * y_pos)

        # This is a constant for this particle
        distance_z = z_pos - z_slice
        distance_z_2 = distance_z * distance_z

        # SWIFT stores hsml as the FWHM.
        kernel_width = kernel_gamma * hsml

        # The number of cells that this kernel spans
        cells_spanned = int32(1.0 + kernel_width * float_res)

        # Now we loop over the square of cells that the kernel lives in
        for cell_x in range(
            # Ensure that the lowest x value is 0, otherwise we'll segfault
            max(0, particle_cell_x - cells_spanned),
            # Ensure that the highest x value lies within the array bounds,
            # otherwise we'll segfault (oops).
            min(particle_cell_x + cells_spanned, maximal_array_index),
        ):
            # The distance in x to our new favourite cell -- remember that our x, y
            # are all in a box of [0, 1]; calculate the distance to the cell centre
            distance_x = (float32(cell_x) + 0.5) * pixel_width - float32(x_pos)
            distance_x_2 = distance_x * distance_x
            for cell_y in range(
                max(0, particle_cell_y - cells_spanned),
                min(particle_cell_y + cells_spanned, maximal_array_index),
            ):
                distance_y = (float32(cell_y) + 0.5) * pixel_width - float32(y_pos)
                distance_y_2 = distance_y * distance_y

                r = sqrt(distance_x_2 + distance_y_2 + distance_z_2)

                kernel_eval = kernel(r, kernel_width)

                image[cell_x, cell_y] += mass * kernel_eval

    return image


def slice_gas_pixel_grid(
    data: SWIFTDataset,
    resolution: int,
    slice: float,
    project: Union[str, None] = "masses",
):
    r"""
    Creates a 2D slice of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return:
        \bar{T} = \sum_j T_j W_{ij}
    ).

    The 'slice' is given in the z direction as a ratio of the boxsize.
    
    Default projection variable is mass. If it is None, then we don't
    weight.

    Creates a resolution x resolution array and returns it, without appropriate
    units.
    """

    if slice > 1.0 or slice < 0.0:
        raise ValueError("Please enter a slice value between 0.0 and 1.0 in slice_gas.")

    number_of_gas_particles = data.gas.particle_ids.size

    if project is None:
        m = ones(number_of_gas_particles, dtype=float32)
    else:
        m = getattr(data.gas, project).value

    box_x, box_y, box_z = data.metadata.boxsize

    # Let's just hope that the box is square otherwise we're probably SOL
    x, y, z = data.gas.coordinates.T
    hsml = data.gas.smoothing_length

    image = slice_scatter(
        x / box_x, y / box_y, z / box_z, m, hsml / box_x, slice, resolution
    )

    return image


def slice_gas(
    data: SWIFTDataset,
    resolution: int,
    slice: float,
    project: Union[str, None] = "masses",
):
    r"""
    Creates a 2D projection of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return:
        \bar{T} = \sum_j T_j W_{ij}
    ).
    
    The 'slice' is given in the z direction as a ratio of the boxsize.

    Default projection variable is mass. If it is None, then we don't
    weight.

    Creates a resolution x resolution array and returns it, with appropriate
    units.
    """

    image = slice_gas_pixel_grid(data, resolution, slice, project)

    units = 1.0 / (
        data.metadata.boxsize[0] * data.metadata.boxsize[1] * data.metadata.boxsize[2]
    )

    if project is not None:
        units *= getattr(data.gas, project).units

    return image * units
