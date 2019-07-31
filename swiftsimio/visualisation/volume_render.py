"""
Basic volume render for SPH data. This takes the 3D positions
of the particles and projects them onto a grid.
"""
from typing import Union
from math import sqrt
from numpy import float64, float32, int32, zeros, array, arange, ndarray, ones
from swiftsimio import SWIFTDataset

from swiftsimio.accelerated import jit

from .slice import kernel, kernel_constant, kernel_gamma


@jit(nopython=True, fastmath=True, parallel=True)
def scatter(
    x: float64, y: float64, z: float64, m: float32, h: float32, res: int
) -> ndarray:
    """
    Creates a scatter plot of:

    + x: the x-positions of the particles. Must be bounded by [0, 1].
    + y: the y-positions of the particles. Must be bounded by [0, 1].
    + z: the y-positions of the particles. Must be bounded by [0, 1].
    + m: the masses (or otherwise weights) of the particles
    + h: the smoothing lengths of the particles
    + res: the number of voxels along one axis, i.e. this returns a cube
           of res * res * res..
    
    This ignores boundary effects.

    Note that explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.
    """
    # Output array for our image
    image = zeros((res, res, res), dtype=float32)
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
    inverse_cell_volume = res * res * res

    for x_pos, y_pos, z_pos, mass, hsml in zip(x, y, z, m, h):
        # Calculate the cell that this particle; use the 64 bit version of the
        # resolution as this is the same type as the positions
        particle_cell_x = int32(float_res_64 * x_pos)
        particle_cell_y = int32(float_res_64 * y_pos)
        particle_cell_z = int32(float_res_64 * z_pos)

        # SWIFT stores hsml as the FWHM.
        kernel_width = kernel_gamma * hsml

        if kernel_width < drop_to_single_cell:
            # Easygame, gg
            image[particle_cell_x, particle_cell_y, particle_cell_z] += (
                mass * inverse_cell_volume
            )
        else:
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
                    for cell_z in range(
                        max(0, particle_cell_z - cells_spanned),
                        min(particle_cell_z + cells_spanned, maximal_array_index),
                    ):
                        distance_z = (float32(cell_z) + 0.5) * pixel_width - float32(
                            z_pos
                        )
                        distance_z_2 = distance_z * distance_z

                        r = sqrt(distance_x_2 + distance_y_2 + distance_z_2)

                        kernel_eval = kernel(r, kernel_width)

                        image[cell_x, cell_y, cell_z] += mass * kernel_eval

    return image


def render_gas_voxel_grid(
    data: SWIFTDataset, resolution: int, project: Union[str, None] = "masses"
):
    r"""
    Creates a 3D projection of a SWIFT dataset, projected by the "project"
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

    box_x, box_y, box_z = data.metadata.boxsize

    # Let's just hope that the box is square otherwise we're probably SOL
    x, y, z = data.gas.coordinates.T
    hsml = data.gas.smoothing_lengths

    image = scatter(x / box_x, y / box_y, z / box_z, m, hsml / box_x, resolution)

    return image


def render_gas(
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

    image = render_gas_voxel_grid(data, resolution, project)

    units = 1.0 / (
        data.metadata.boxsize[0] * data.metadata.boxsize[1] * data.metadata.boxsize[2]
    )

    if project is not None:
        units *= getattr(data.gas, project).units

    return image * units
