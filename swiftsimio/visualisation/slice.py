"""
Sub-module for slice plots in SWFITSIMio.
"""

from typing import Union
from math import sqrt
from numpy import float64, float32, int32, zeros, array, arange, ndarray, ones, isclose
from unyt import unyt_array
from swiftsimio import SWIFTDataset

from swiftsimio.accelerated import jit, prange, NUM_THREADS

# Taken from Dehnen & Aly 2012
kernel_gamma = 1.936492
kernel_constant = 21.0 * 0.31830988618379067154 / 2.0


@jit(nopython=True, fastmath=True)
def kernel(r: Union[float, float32], H: Union[float, float32]):
    r"""
    Kernel implementation for swiftsimio. 

    Parameters
    ----------
    r : float or float32
        Distance from particle

    H : float or float32
        Kernel width (i.e. radius of compact support of kernel)

    Returns
    -------
    float
        Contribution to density by particle at distance `r`

    Notes
    -----
    Swiftsimio uses the Wendland-C2 kernel as described in [1]_.

    References
    ----------
    .. [1] Denhen & Aly (2012) ALEXEI: add proper citation

    """
    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = 0.0

    if ratio < 1.0:
        one_minus_ratio = 1.0 - ratio
        one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
        one_minus_ratio_4 = one_minus_ratio_2 * one_minus_ratio_2

        kernel = max(one_minus_ratio_4 * (1.0 + 4.0 * ratio), 0.0)

        kernel *= kernel_constant * inverse_H * inverse_H * inverse_H

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
    r"""
    Creates a scatter plot of the given quantities for a particles in a data slice ignoring boundary effects.

    Parameters
    ----------
    x : array of float64
        x-positions of the particles. Must be bounded by [0, 1].
    y : array of float64
        y-positions of the particles. Must be bounded by [0, 1].
    z : array of float64
        z-positions of the particles. Must be bounded by [0, 1].
    m : array of float32
        masses (or otherwise weights) of the particles
    h : array of float32
        smoothing lengths of the particles
    z_slice : float64
        the position at which we wish to create the slice
    res : int
        the number of pixels.

    Returns
    -------
    ndarray of float32
        output array for scatterplot image
    
    Notes
    -----
    Explicitly defining the types in this function allows
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

        if (
            # No overlap in z
            distance_z_2 > (kernel_width * kernel_width)
            # No overlap in x, y
            or particle_cell_x + cells_spanned < 0
            or particle_cell_x - cells_spanned > maximal_array_index
            or particle_cell_y + cells_spanned < 0
            or particle_cell_y - cells_spanned > maximal_array_index
        ):
            # We have no overlap, we can skip this particle.
            continue

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


@jit(nopython=True, fastmath=True, parallel=True)
def slice_scatter_parallel(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float64,
    res: int,
) -> ndarray:
    r"""
    Parallel implementation of slice_scatter
    
    Creates a scatter plot of the given quantities for a particles in a data slice ignoring boundary effects.

    Parameters
    ----------
    x : array of float64
        x-positions of the particles. Must be bounded by [0, 1].
    y : array of float64
        y-positions of the particles. Must be bounded by [0, 1].
    z : array of float64
        z-positions of the particles. Must be bounded by [0, 1].
    m : array of float32
        masses (or otherwise weights) of the particles
    h : array of float32
        smoothing lengths of the particles
    z_slice : float64
        the position at which we wish to create the slice
    res : int
        the number of pixels.

    Returns
    -------
    ndarray of float32
        output array for scatterplot image
    
    Notes
    -----
    ALEXEI: Check with Josh, do these notes still hold?
    Explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.
    """
    #Same as scatter, but executes in parallel! This is actually trivial,
    #we just make NUM_THREADS images and add them together at the end.

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

        output += slice_scatter(
            x=x[left_edge:right_edge],
            y=y[left_edge:right_edge],
            z=z[left_edge:right_edge],
            m=m[left_edge:right_edge],
            h=h[left_edge:right_edge],
            z_slice=z_slice,
            res=res,
        )

    return output


def slice_gas_pixel_grid(
    data: SWIFTDataset,
    resolution: int,
    slice: float,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    region: Union[None, unyt_array] = None,
):
    r"""
    Creates a 2D slice of a SWIFT dataset, weighted by data field, in the
    form of a pixel grid.

    Parameters
    ----------
    data : SWIFTDataset
        Dataset from which slice is extracted

    resolution : int
        Specifies size of return array
        
    slice : float
        Specifies the location along the z-axis where the slice is to be
        extracted as a fraction of boxsize.

    project : str, optional
        Data field to be projected. Default is mass. If None then simply
        count number of particles (ALEXEI: check wording with Josh)
    
    parallel : bool
        used to determine if we will create the image in parallel. This 
        defaults to False, but can speed up the creation of large images 
        significantly at the cost of increased memory usage.

    region : array, optional
        determines where the image will be created
        (this corresponds to the left and right-hand edges, and top and bottom edges)
        if it is not None. It should have a length of four, and take the form:

        [x_min, x_max, y_min, y_max]

        Particles outside of this range are still considered if their
        smoothing lengths overlap with the range.

    Returns
    -------
    ndarray of float32
        Creates a `resolution` x `resolution` array and returns it, without appropriate
        units.

    """

    if slice > 1.0 or slice < 0.0:
        raise ValueError("Please enter a slice value between 0.0 and 1.0 in slice_gas.")

    number_of_gas_particles = data.gas.coordinates.shape[0]

    if project is None:
        m = ones(number_of_gas_particles, dtype=float32)
    else:
        m = getattr(data.gas, project).value

    box_x, box_y, box_z = data.metadata.boxsize

    # Set the limits of the image.
    if region is not None:
        x_min, x_max, y_min, y_max = region
    else:
        x_min = (0 * box_x).to(box_x.units)
        x_max = box_x
        y_min = (0 * box_y).to(box_y.units)
        y_max = box_y

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Test that we've got a square box
    if not isclose(x_range.value, y_range.value):
        raise AttributeError(
            "Slice code is currently not able to handle non-square images"
        )

    x, y, z = data.gas.coordinates.T

    try:
        hsml = data.gas.smoothing_lengths
    except AttributeError:
        # Backwards compatibility
        hsml = data.gas.smoothing_length

    common_parameters = dict(
        x=(x - x_min) / x_range,
        y=(y - y_min) / y_range,
        z=z / box_z,
        m=m,
        h=hsml / x_range,
        z_slice=slice,
        res=resolution,
    )

    if parallel:
        image = slice_scatter_parallel(**common_parameters)
    else:
        image = slice_scatter(**common_parameters)

    return image


def slice_gas(
    data: SWIFTDataset,
    resolution: int,
    slice: float,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    region: Union[None, unyt_array] = None,
):
    r"""
    Creates a 2D slice of a SWIFT dataset, weighted by data field

    Parameters
    ----------
    data : SWIFTDataset
        Dataset from which slice is extracted

    resolution : int
        Specifies size of return array
        
    slice : float
        Specifies the location along the z-axis where the slice is to be
        extracted as a fraction of boxsize.

    project : str, optional
        Data field to be projected. Default is mass. If None then simply
        count number of particles (ALEXEI: check wording with Josh)
    
    parallel : bool
        used to determine if we will create the image in parallel. This 
        defaults to False, but can speed up the creation of large images 
        significantly at the cost of increased memory usage.

    region : array, optional
        determines where the image will be created
        (this corresponds to the left and right-hand edges, and top and bottom edges)
        if it is not None. It should have a length of four, and take the form:

        [x_min, x_max, y_min, y_max]

        Particles outside of this range are still considered if their
        smoothing lengths overlap with the range.

    Returns
    -------
    ndarray of float32
        Creates a `resolution` x `resolution` array and returns it, without appropriate
        units.

    Notes
    -----
    This is a wrapper function for slice_gas_pixel_grid ensuring that output units are
    appropriate
    """

    image = slice_gas_pixel_grid(data, resolution, slice, project, parallel, region)

    if region is not None:
        x_range = region[1] - region[0]
        y_range = region[3] - region[2]
        units = 1.0 / (x_range * y_range * data.metadata.boxsize[2])
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(
            1.0 / (x_range.units * y_range.units * data.metadata.boxsize.units)
        )
    else:
        units = 1.0 / (
            data.metadata.boxsize[0]
            * data.metadata.boxsize[1]
            * data.metadata.boxsize[2]
        )
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / data.metadata.boxsize.units ** 3)

    if project is not None:
        units *= getattr(data.gas, project).units

    return image * units
