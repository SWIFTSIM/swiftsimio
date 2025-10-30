"""
Renormalised projection visualisation.

This version of the function is the same as `fast` but provides an explicit
renormalisation of each kernel such that the mass is conserved up to floating point
precision.

This is the original smoothing code. This provides basic renormalisation of the kernel on
each call.
"""

import numpy as np
from swiftsimio.accelerated import jit, NUM_THREADS, prange

from swiftsimio.visualisation.projection_backends.kernels import (
    kernel_single_precision as kernel,
)
from swiftsimio.visualisation.projection_backends.kernels import kernel_gamma


@jit(nopython=True, fastmath=True)
def scatter(
    x: np.float64,
    y: np.float64,
    m: np.float32,
    h: np.float32,
    res: int,
    box_x: np.float64 = 0.0,
    box_y: np.float64 = 0.0,
) -> np.ndarray:
    """
    Create a weighted scatter plot.

    Computes contributions to from particles with positions
    (`x`,`y`) with smoothing lengths `h` weighted by quantities `m`.
    This includes periodic boundary effects.

    Parameters
    ----------
    x : np.ndarray[np.float64]
        Array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.ndarray[np.float64]
        Array of y-positions of the particles. Must be bounded by [0, 1].

    m : np.ndarray[np.float32]
        Array of masses (or otherwise weights) of the particles.

    h : np.ndarray[np.float32]
        Array of smoothing lengths of the particles.

    res : int
        The number of pixels along one axis, i.e. this returns a square
        of res * res.

    box_x : np.float64
        Box size in x, in the same rescaled length units as x and y. Used
        for periodic wrapping.

    box_y : np.float64
        Box size in y, in the same rescaled length units as x and y. Used
        for periodic wrapping.

    Returns
    -------
    np.ndarray[np.float32, np.float32, np.float32]
        Pixel grid of quantity.

    See Also
    --------
    scatter_parallel
        Parallel implementation of this function.

    Notes
    -----
    Explicitly defining the types in this function allows for a 25-50% performance
    improvement. In our testing, using numpy floats and integers is also an improvement
    over using the numba ones.
    """
    # Output array for our image
    image = np.zeros((res, res), dtype=np.float32)
    maximal_array_index = np.int32(res) - 1

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    float_res = np.float32(res)
    pixel_width = 1.0 / float_res

    # We need this for combining with the x_pos and y_pos variables.
    float_res_64 = np.float64(res)

    # Pre-calculate this constant for use with the above
    inverse_cell_area = float_res * float_res

    if box_x == 0.0:
        xshift_min = 0
        xshift_max = 1
    else:
        xshift_min = -1  # x_min is always at x=0
        xshift_max = int(np.ceil(1 / box_x) + 1)  # tile the box to cover [0, 1]
    if box_y == 0.0:
        yshift_min = 0
        yshift_max = 1
    else:
        yshift_min = -1  # y_min is always at y=0
        yshift_max = int(np.ceil(1 / box_y) + 1)  # tile the box to cover [0, 1]

    for x_pos_original, y_pos_original, mass, hsml in zip(x, y, m, h):
        # loop over periodic copies of this particle
        for xshift in range(xshift_min, xshift_max):
            for yshift in range(yshift_min, yshift_max):
                x_pos = x_pos_original + xshift * box_x
                y_pos = y_pos_original + yshift * box_y

                # Calculate the cell that this particle; use the 64 bit version of the
                # resolution as this is the same type as the positions
                particle_cell_x = np.int32(np.floor(float_res_64 * x_pos))
                particle_cell_y = np.int32(np.floor(float_res_64 * y_pos))

                # SWIFT stores hsml as the FWHM.
                kernel_width = kernel_gamma * hsml

                # The number of cells that this kernel spans
                cells_spanned = np.int32(1.0 + kernel_width * float_res)

                if (
                    particle_cell_x + cells_spanned < 0
                    or particle_cell_x - cells_spanned > maximal_array_index
                    or particle_cell_y + cells_spanned < 0
                    or particle_cell_y - cells_spanned > maximal_array_index
                ):
                    # Can happily skip this particle
                    continue

                if cells_spanned <= 1:
                    # Easygame, gg
                    if (
                        particle_cell_x >= 0
                        and particle_cell_x <= maximal_array_index
                        and particle_cell_y >= 0
                        and particle_cell_y <= maximal_array_index
                    ):
                        image[particle_cell_x, particle_cell_y] += (
                            mass * inverse_cell_area
                        )
                else:
                    # First calculate the typical normalisation for this kernel in this
                    # segment of the image (use the whole square, including overlap with
                    # edges).

                    normalisation = np.float32(0.0)

                    for cell_x in range(
                        particle_cell_x - cells_spanned + 1,
                        particle_cell_x + cells_spanned,
                    ):
                        # The distance in x to our new favourite cell -- remember that our
                        # x, y are all in a box of [0, 1]; calculate the distance to the
                        # cell centre
                        distance_x = (
                            np.float32(cell_x) + 0.5
                        ) * pixel_width - np.float32(x_pos)
                        distance_x_2 = distance_x * distance_x
                        for cell_y in range(
                            particle_cell_y - cells_spanned + 1,
                            particle_cell_y + cells_spanned,
                        ):
                            distance_y = (
                                np.float32(cell_y) + 0.5
                            ) * pixel_width - np.float32(y_pos)
                            distance_y_2 = distance_y * distance_y

                            r = np.sqrt(distance_x_2 + distance_y_2)

                            normalisation += kernel(r, kernel_width)

                    # We want sum(W_ij) = 1.0
                    normalisation = inverse_cell_area / normalisation

                    # Now we loop over the square of cells that the kernel lives in
                    for cell_x in range(
                        # Ensure that the lowest x value is 0, otherwise we'll segfault
                        max(0, particle_cell_x - cells_spanned),
                        # Ensure that the highest x value lies within the array bounds,
                        # otherwise we'll segfault (oops).
                        min(
                            particle_cell_x + cells_spanned + 1, maximal_array_index + 1
                        ),
                    ):
                        # The distance in x to our new favourite cell -- remember that our
                        # x, y are all in a box of [0, 1]; calculate the distance to the
                        # cell centre
                        distance_x = (
                            np.float32(cell_x) + 0.5
                        ) * pixel_width - np.float32(x_pos)
                        distance_x_2 = distance_x * distance_x
                        for cell_y in range(
                            max(0, particle_cell_y - cells_spanned),
                            min(
                                particle_cell_y + cells_spanned + 1,
                                maximal_array_index + 1,
                            ),
                        ):
                            distance_y = (
                                np.float32(cell_y) + 0.5
                            ) * pixel_width - np.float32(y_pos)
                            distance_y_2 = distance_y * distance_y

                            r = np.sqrt(distance_x_2 + distance_y_2)

                            # Renormalise kernel before using
                            kernel_eval = normalisation * kernel(r, kernel_width)

                            image[cell_x, cell_y] += mass * kernel_eval

    return image


@jit(nopython=True, fastmath=True, parallel=True)
def scatter_parallel(
    x: np.float64,
    y: np.float64,
    m: np.float32,
    h: np.float32,
    res: int,
    box_x: np.float64 = 0.0,
    box_y: np.float64 = 0.0,
) -> np.ndarray:
    """
    Parallel implementation of scatter.

    Creates a weighted scatter plot. Computes contributions from
    particles with positions (`x`,`y`) with smoothing lengths `h`
    weighted by quantities `m`.
    This includes periodic boundary effects.

    Parameters
    ----------
    x : np.ndarray[np.float64]
        Array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.ndarray[np.float64]
        Array of y-positions of the particles. Must be bounded by [0, 1].

    m : np.ndarray[np.float32]
        Array of masses (or otherwise weights) of the particles.

    h : np.ndarray[np.float32]
        Array of smoothing lengths of the particles.

    res : int
        The number of pixels along one axis, i.e. this returns a square
        of res * res.

    box_x : np.float64
        Box size in x, in the same rescaled length units as x and y. Used
        for periodic wrapping.

    box_y : np.float64
        Box size in y, in the same rescaled length units as x and y. Used
        for periodic wrapping.

    Returns
    -------
    np.ndarray[np.float32, np.float32, np.float32]
        Pixel grid of quantity.

    See Also
    --------
    scatter
        Creates 2D scatter plot from SWIFT data.

    Notes
    -----
    Explicitly defining the types in this function allows for a 25-50% performance
    improvement. In our testing, using numpy floats and integers is also an improvement
    over using the numba ones.
    """
    # Same as scatter, but executes in parallel! This is actually trivial,
    # we just make NUM_THREADS images and add them together at the end.

    number_of_particles = x.size
    core_particles = number_of_particles // NUM_THREADS

    output = np.zeros((res, res), dtype=np.float32)

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
            box_x=box_x,
            box_y=box_y,
        )

    return output
