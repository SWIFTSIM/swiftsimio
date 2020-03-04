"""
Contains visualisation functions that are accelerated with numba
if it is available.
"""

from typing import Union
from math import sqrt
from numpy import (
    float64,
    float32,
    int32,
    zeros,
    array,
    arange,
    ndarray,
    ones,
    isclose,
    matmul,
)
from unyt import unyt_array, unyt_quantity
from swiftsimio import SWIFTDataset

from swiftsimio.reader import __SWIFTParticleDataset
from swiftsimio.accelerated import jit, NUM_THREADS, prange

# Taken from Dehnen & Aly 2012
kernel_gamma = 1.897367
kernel_constant = 7.0 / 3.14159


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
        one_minus_ratio = 1.0 - ratio
        one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
        one_minus_ratio_4 = one_minus_ratio_2 * one_minus_ratio_2

        kernel = max(one_minus_ratio_4 * (1.0 + 4.0 * ratio), 0.0)

        kernel *= kernel_constant * inverse_H * inverse_H

    return kernel


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

    for x_pos, y_pos, mass, hsml in zip(x, y, m, h):
        # Calculate the cell that this particle; use the 64 bit version of the
        # resolution as this is the same type as the positions
        particle_cell_x = int32(float_res_64 * x_pos)
        particle_cell_y = int32(float_res_64 * y_pos)

        # SWIFT stores hsml as the FWHM.
        kernel_width = kernel_gamma * hsml

        # The number of cells that this kernel spans
        cells_spanned = int32(1.0 + kernel_width * float_res)

        if (
            particle_cell_x + cells_spanned < 0
            or particle_cell_x - cells_spanned > maximal_array_index
            or particle_cell_y + cells_spanned < 0
            or particle_cell_y - cells_spanned > maximal_array_index
        ):
            # Can happily skip this particle
            continue

        if kernel_width < drop_to_single_cell:
            # Easygame, gg
            image[particle_cell_x, particle_cell_y] += mass * inverse_cell_area
        else:
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

                    r = sqrt(distance_x_2 + distance_y_2)

                    kernel_eval = kernel(r, kernel_width)

                    image[cell_x, cell_y] += mass * kernel_eval

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

    output = zeros((res, res), dtype=float32)

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


def project_pixel_grid(
    data: __SWIFTParticleDataset,
    boxsize: unyt_array,
    resolution: int,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    region: Union[None, unyt_array] = None,
    rotation_matrix: Union[None, array] = None,
    rotation_center: Union[None, unyt_array] = None,
):
    r"""
    Creates a 2D projection of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return: \bar{T} = \sum_j T_j
    W_{ij}).

    Differs from its particle-propery named counterparts as this uses
    a particledataset instead of the full dataset.

    Default projection variable is mass. If it is None, then we don't
    weight.

    Creates a resolution x resolution array and returns it, without appropriate
    units.

    The parallel argument, is used to determine if we will create the image
    in parallel. This defaults to False, but can speed up the creation of large
    images significantly.

    The final argument, region, determines where the image will be created
    (this corresponds to the left and right-hand edges, and top and bottom edges)
    if it is not None. It should have a length of four, and take the form:

        [x_min, x_max, y_min, y_max]

    Note that particles outside of this range are still considered if their
    smoothing lengths overlap with the range.
    """

    number_of_particles = data.coordinates.shape[0]

    if project is None:
        m = ones(number_of_particles, dtype=float32)
    else:
        m = getattr(data, project).value

    box_x, box_y, _ = boxsize

    # Set the limits of the image.
    if region is not None:
        x_min, x_max, y_min, y_max = region
    else:
        x_min = unyt_quantity(0.0, units=box_x.units)
        x_max = box_x
        y_min = unyt_quantity(0.0, units=box_y.units)
        y_max = box_y

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Test that we've got a square box
    if not isclose(x_range.value, y_range.value):
        raise AttributeError(
            "Projection code is currently not able to handle non-square images"
        )

    try:
        hsml = data.smoothing_lengths
    except AttributeError:
        # Backwards compatibility
        hsml = data.smoothing_length

    if rotation_center is not None:
        # Rotate co-ordinates as required
        x, y, _ = matmul(rotation_matrix, (data.coordinates - rotation_center).T)

        x += rotation_center[0]
        y += rotation_center[1]

    else:
        x, y, _ = data.coordinates.T

    common_arguments = dict(
        x=(x - x_min) / x_range,
        y=(y - y_min) / y_range,
        m=m,
        h=hsml / x_range,
        res=resolution,
    )

    if parallel:
        image = scatter_parallel(**common_arguments)
    else:
        image = scatter(**common_arguments)

    return image


def project_gas_pixel_grid(
    data: SWIFTDataset,
    resolution: int,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    region: Union[None, unyt_array] = None,
    rotation_matrix: Union[None, array] = None,
    rotation_center: Union[None, unyt_array] = None,
):
    r"""
    Creates a 2D projection of a SWIFT dataset, projected by the "project"
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
    (this corresponds to the left and right-hand edges, and top and bottom edges)
    if it is not None. It should have a length of four, and take the form:

        [x_min, x_max, y_min, y_max]

    Note that particles outside of this range are still considered if their
    smoothing lengths overlap with the range.
    """

    image = project_pixel_grid(
        data=data.gas,
        boxsize=data.metadata.boxsize,
        resolution=resolution,
        project=project,
        parallel=parallel,
        region=region,
        rotation_matrix=rotation_matrix,
        rotation_center=rotation_center,
    )

    return image


def project_gas(
    data: SWIFTDataset,
    resolution: int,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    region: Union[None, unyt_array] = None,
    rotation_matrix: Union[None, array] = None,
    rotation_center: Union[None, unyt_array] = None,
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
    (this corresponds to the left and right-hand edges, and top and bottom edges)
    if it is not None. It should have a length of four, and take the form:

        [x_min, x_max, y_min, y_max]

    Note that particles outside of this range are still considered if their
    smoothing lengths overlap with the range.
    """

    image = project_gas_pixel_grid(
        data, resolution, project, parallel, region, rotation_matrix, rotation_center
    )

    if region is not None:
        x_range = region[1] - region[0]
        y_range = region[3] - region[2]
        units = 1.0 / (x_range * y_range)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / (x_range.units * y_range.units))
    else:
        units = 1.0 / (data.metadata.boxsize[0] * data.metadata.boxsize[1])
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / data.metadata.boxsize.units ** 2)

    if project is not None:
        units *= getattr(data.gas, project).units

    return image * units
