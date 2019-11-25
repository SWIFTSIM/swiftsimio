"""
Basic volume render for SPH data. This takes the 3D positions
of the particles and projects them onto a grid.
"""
from typing import Union
from math import sqrt
from numpy import float64, float32, int32, zeros, array, arange, ndarray, ones, isclose
from unyt import unyt_array
from swiftsimio import SWIFTDataset

from swiftsimio.accelerated import jit, NUM_THREADS, prange

from .slice import kernel, kernel_constant, kernel_gamma


@jit(nopython=True, fastmath=True)
def scatter(
    x: float64, y: float64, z: float64, m: float32, h: float32, res: int
) -> ndarray:
    """
    Creates a voxel grid of:

    + x: the x-positions of the particles. Must be bounded by [0, 1].
    + y: the y-positions of the particles. Must be bounded by [0, 1].
    + z: the z-positions of the particles. Must be bounded by [0, 1].
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


@jit(nopython=True, fastmath=True, parallel=True)
def scatter_parallel(
    x: float64, y: float64, z: float64, m: float32, h: float32, res: int
) -> ndarray:
    """
    Same as scatter, but executes in parallel! This is actually trivial,
    we just make NUM_THREADS images and add them together at the end.
    """

    number_of_particles = x.size
    core_particles = number_of_particles // NUM_THREADS

    output = zeros((res, res, res), dtype=float32)

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
            z=z[left_edge:right_edge],
            m=m[left_edge:right_edge],
            h=h[left_edge:right_edge],
            res=res,
        )

    return output


def render_gas_voxel_grid(
    data: SWIFTDataset,
    resolution: int,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    region: Union[None, unyt_array] = None,
):
    r"""
    Creates a 3D projection of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return: \bar{T} = \sum_j T_j
    W_{ij}).
    
    Default projection variable is mass. If it is None, then we don't
    weight.

    Creates a resolution x resolution array and returns it, without appropriate
    units.

    The parallel argument, is used to determine if we will create the image
    in parallel. This defaults to False, but can speed up the creation of large
    images significantly.

    The final argument, region, determines where the image will be created
    (this corresponds to the left and right-hand edges, and top and bottom
    edges, and front and back edges) if it is not None. It should have a
    length of six, and take the form:

        [x_min, x_max, y_min, y_max, z_min, z_max]

    Note that particles outside of this range are still considered if their
    smoothing lengths overlap with the range.
    """

    number_of_gas_particles = data.gas.particle_ids.size

    if project is None:
        m = ones(number_of_gas_particles, dtype=float32)
    else:
        m = getattr(data.gas, project).value

    box_x, box_y, box_z = data.metadata.boxsize

    # Set the limits of the image.
    if region is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = region
    else:
        x_min = 0 * box_x
        x_max = box_x
        y_min = 0 * box_y
        y_max = box_y
        z_min = 0 * box_z
        z_max = box_z

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Test that we've got a cubic box
    if not (
        isclose(x_range.value, y_range.value) and isclose(x_range.value, z_range.value)
    ):
        raise AttributeError(
            "Projection code is currently not able to handle non-cubic images"
        )

    # Let's just hope that the box is square otherwise we're probably SOL
    x, y, z = data.gas.coordinates.T
    hsml = data.gas.smoothing_lengths

    arguments = dict(
        x=(x - x_min) / x_range,
        y=(y - y_min) / y_range,
        z=(z - z_min) / z_range,
        m=m,
        h=hsml / x_range,
        res=resolution,
    )

    if parallel:
        image = scatter_parallel(**arguments)
    else:
        image = scatter(**arguments)

    return image


def render_gas(
    data: SWIFTDataset,
    resolution: int,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    region: Union[None, unyt_array] = None,
):
    r"""
    Creates a 2D projection of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return: \bar{T} = \sum_j T_j
    W_{ij}).
    
    Default projection variable is mass. If it is None, then we don't
    weight.

    Creates a resolution x resolution array and returns it, with appropriate
    units.

    The parallel argument, is used to determine if we will create the image
    in parallel. This defaults to False, but can speed up the creation of large
    images significantly.

    The final argument, region, determines where the image will be created
    (this corresponds to the left and right-hand edges, and top and bottom
    edges, and front and back edges) if it is not None. It should have a
    length of six, and take the form:

        [x_min, x_max, y_min, y_max, z_min, z_max]

    Note that particles outside of this range are still considered if their
    smoothing lengths overlap with the range.
    """

    image = render_gas_voxel_grid(data, resolution, project, parallel, region=region)

    if region is not None:
        x_range = region[1] - region[0]
        y_range = region[3] - region[2]
        z_range = region[5] - region[4]
        units = 1.0 / (x_range * y_range * z_range)
    else:
        units = 1.0 / (
            data.metadata.boxsize[0]
            * data.metadata.boxsize[1]
            * data.metadata.boxsize[2]
        )

    if project is not None:
        units *= getattr(data.gas, project).units

    return image * units
