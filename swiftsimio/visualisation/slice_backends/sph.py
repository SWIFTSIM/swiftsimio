from typing import Union
from math import sqrt
from numpy import float64, float32, int32, zeros, ndarray

from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.accelerated import jit, prange, NUM_THREADS

# Taken from Dehnen & Aly 2012
kernel_gamma = 1.936492
kernel_constant = 21.0 * 0.31830988618379067154 / 2.0


@jit(nopython=True, fastmath=True)
def kernel(r: Union[float, float32], H: Union[float, float32]):
    """
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
    .. [1] Dehnen W., Aly H., 2012, MNRAS, 425, 1068

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


def get_hsml(data: SWIFTDataset) -> cosmo_array:
    """
    Extract the smoothing lengths from the gas particles (used for slicing).

    Parameters
    ----------
    data : SWIFTDataset
        The Dataset from which slice will be extracted

    Returns
    -------
    The extracted smoothing lengths.
    """
    try:
        hsml = data.gas.smoothing_lengths
    except AttributeError:
        # Backwards compatibility
        hsml = data.gas.smoothing_length
    return hsml


@jit(nopython=True, fastmath=True)
def slice_scatter(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float64,
    res: int,
    box_x: float64 = 0.0,
    box_y: float64 = 0.0,
    box_z: float64 = 0.0,
) -> ndarray:
    """
    Creates a scatter plot of the given quantities for a particles in a data slice including periodic boundary effects.

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
    xres : int
        the number of pixels in the x-direction.
    yres : int
        the number of pixels in the y-direction.
    box_x: float64
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.
    box_y: float64
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.
    box_z: float64
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.

    Returns
    -------
    ndarray of float32
        output array for scatterplot image

    See Also
    --------
    scatter : Create 3D scatter plot of SWIFT data
    scatter_parallel : Create 3D scatter plot of SWIFT data in parallel
    slice_scatter_parallel : Create scatter plot of a slice of data in parallel

    Notes
    -----
    Explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.
    """
    # Output array for our image
    res = int(max(xres, yres))
    image = zeros((res, res), dtype=float32)
    maximal_array_index = int32(res) - 1

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    float_res = float32(res)
    pixel_width = 1.0 / float_res

    # We need this for combining with the x_pos and y_pos variables.
    float_res_64 = float64(res)

    if box_x == 0.0:
        xshift_min = 0
        xshift_max = 1
    else:
        xshift_min = -1
        xshift_max = 2
    if box_y == 0.0:
        yshift_min = 0
        yshift_max = 1
    else:
        yshift_min = -1
        yshift_max = 2
    if box_z == 0.0:
        zshift_min = 0
        zshift_max = 1
    else:
        zshift_min = -1
        zshift_max = 2

    for x_pos_original, y_pos_original, z_pos_original, mass, hsml in zip(
        x, y, z, m, h
    ):
        # loop over periodic copies of the particle
        for xshift in range(xshift_min, xshift_max):
            for yshift in range(yshift_min, yshift_max):
                for zshift in range(zshift_min, zshift_max):
                    x_pos = x_pos_original + xshift * box_x
                    y_pos = y_pos_original + yshift * box_y
                    z_pos = z_pos_original + zshift * box_z

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
                        min(particle_cell_x + cells_spanned, maximal_array_index + 1),
                    ):
                        # The distance in x to our new favourite cell -- remember that our x, y
                        # are all in a box of [0, 1]; calculate the distance to the cell centre
                        distance_x = (float32(cell_x) + 0.5) * pixel_width - float32(
                            x_pos
                        )
                        distance_x_2 = distance_x * distance_x
                        for cell_y in range(
                            max(0, particle_cell_y - cells_spanned),
                            min(
                                particle_cell_y + cells_spanned, maximal_array_index + 1
                            ),
                        ):
                            distance_y = (
                                float32(cell_y) + 0.5
                            ) * pixel_width - float32(y_pos)
                            distance_y_2 = distance_y * distance_y

                            r = sqrt(distance_x_2 + distance_y_2 + distance_z_2)

                            kernel_eval = kernel(r, kernel_width)

                            image[cell_x, cell_y] += mass * kernel_eval

    # trim the image to remove empty pixels
    return image[:xres, :yres]


@jit(nopython=True, fastmath=True, parallel=True)
def slice_scatter_parallel(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float64,
    res: int,
    box_x: float64 = 0.0,
    box_y: float64 = 0.0,
    box_z: float64 = 0.0,
) -> ndarray:
    """
    Parallel implementation of slice_scatter

    Creates a scatter plot of the given quantities for a particles in a data slice including periodic boundary effects.

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
    xres : int
        the number of pixels in the x-direction.
    yres : int
        the number of pixels in the y-direction.
    box_x: float64
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.
    box_y: float64
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.
    box_z: float64
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.

    Returns
    -------
    ndarray of float32
        output array for scatterplot image

    See Also
    --------
    scatter : Create 3D scatter plot of SWIFT data
    scatter_parallel : Create 3D scatter plot of SWIFT data in parallel
    slice_scatter : Create scatter plot of a slice of data

    Notes
    -----
    Explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.
    """
    # Same as scatter, but executes in parallel! This is actually trivial,
    # we just make NUM_THREADS images and add them together at the end.

    number_of_particles = x.size
    core_particles = number_of_particles // NUM_THREADS

    output = zeros((xres, int(yres)), dtype=float32)

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
            xres=xres,
            yres=yres,
            box_x=box_x,
            box_y=box_y,
            box_z=box_z,
        )

    return output
