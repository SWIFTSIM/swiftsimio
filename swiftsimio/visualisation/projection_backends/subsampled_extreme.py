"""
Sub-sampled smoothing kernel with each kernel evaluated
at least 64^2 times. This uses a dithered pre-calculated
kernel for cell overlaps at small scales, and at large
scales uses subsampling.

Uses double precision.
"""


"""
The original smoothing code. This provides a paranoid supersampling
of the kernel.
"""


from typing import Union
from math import sqrt, ceil
from numpy import float32, float64, int32, zeros, ndarray

from swiftsimio.accelerated import jit, NUM_THREADS, prange
from swiftsimio.accelerated import jit, NUM_THREADS, prange
from swiftsimio.visualisation.projection_backends.kernels import (
    kernel_double_precision as kernel,
)
from swiftsimio.visualisation.projection_backends.kernels import (
    kernel_constant,
    kernel_gamma,
)

kernel_constant = float64(kernel_constant)
kernel_gamma = float64(kernel_gamma)


@jit(nopython=True, fastmath=True)
def scatter(x: float64, y: float64, m: float32, h: float32, res: int) -> ndarray:
    """
    Creates a weighted scatter plot

    Computes contributions to from particles with positions
    (`x`,`y`) with smoothing lengths `h` weighted by quantities `m`.
    This ignores boundary effects.

    Parameters
    ----------

    x : np.array[float64]
        array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.array[float64]
        array of y-positions of the particles. Must be bounded by [0, 1].

    m : np.array[float32]
        array of masses (or otherwise weights) of the particles

    h : np.array[float32]
        array of smoothing lengths of the particles

    res : int
        the number of pixels along one axis, i.e. this returns a square
        of res * res.

    Returns
    -------

    np.array[float32, float32, float32]
        pixel grid of quantity

    See Also
    --------

    scatter_parallel : Parallel implementation of this function

    Notes
    -----

    Explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.

    Uses 4x the number of sampling points as in scatter in subsampled.py
    """
    # Output array for our image
    image = zeros((res, res), dtype=float64)
    maximal_array_index = int32(res)

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    float_res = float64(res)
    pixel_width = 1.0 / float_res

    # Pre-calculate this constant for use with the above
    inverse_cell_area = float_res * float_res

    # Minimum number of kernel evaluations for each particle (this x2 squared)
    MIN_KERNEL_EVALUATIONS = 32
    float_MIN_KERNEL_EVALUATIONS = float64(MIN_KERNEL_EVALUATIONS)

    # Dithered kernel evaluations - note it is a 2x DITHER_EVALUATIONS^2 grid
    DITHER_EVALUATIONS = 64
    float_DITHER_EVALUATIONS = float64(DITHER_EVALUATIONS)
    float_DITHER_EVALUATIONS_inv = 1.0 / float_DITHER_EVALUATIONS

    # Pre-comute a min_kernel_evaluations x min_kernel_evaluations square
    # for dithering in cases of small kernel overlap
    dithered_kernel = zeros(
        (2 * DITHER_EVALUATIONS, 2 * DITHER_EVALUATIONS), dtype=float64
    )

    # Fill with kernel evaluations
    for x_dither_cell in range(2 * DITHER_EVALUATIONS):
        x_float = float64(x_dither_cell) + 0.5
        x_dither_distance = x_float - float_DITHER_EVALUATIONS
        x_dither_distance_squared = x_dither_distance * x_dither_distance
        for y_dither_cell in range(2 * DITHER_EVALUATIONS):
            y_float = float64(y_dither_cell) + 0.5
            y_dither_distance = y_float - float_DITHER_EVALUATIONS
            y_dither_distance_squared = y_dither_distance * y_dither_distance

            r = sqrt(x_dither_distance_squared + y_dither_distance_squared)
            dithered_kernel[x_dither_cell, y_dither_cell] += kernel(
                r, H=float_DITHER_EVALUATIONS
            )

    # May as well have this correctly normed.
    dithered_kernel *= inverse_cell_area / dithered_kernel.sum()

    for x_pos, y_pos, mass, hsml in zip(x, y, m, h):
        # Calculate the cell that this particle; use the 64 bit version of the
        # resolution as this is the same type as the positions
        particle_cell_x = int32(float_res * x_pos)
        particle_cell_y = int32(float_res * y_pos)

        # SWIFT stores hsml as the FWHM.
        float_mass = float64(mass)
        kernel_width = float64(kernel_gamma * hsml)

        # The number of cells that this kernel spans
        float_cells_spanned = 1.0 + kernel_width * float_res
        cells_spanned = int32(float_cells_spanned)

        if (
            particle_cell_x + cells_spanned < 0
            or particle_cell_x - cells_spanned > maximal_array_index
            or particle_cell_y + cells_spanned < 0
            or particle_cell_y - cells_spanned > maximal_array_index
        ):
            # Can happily skip this particle
            continue

        # If the particle is too small, then it's very likely that:
        # a) it does not lie on a boundary
        # b) evaluating it over this boundary would cause significant errors
        if kernel_width <= 0.25 * pixel_width:
            # Here we check for overlaps between this kernel and boundaries.
            # If they exist, we must use the sub-sampled kernel.

            dx_left = x_pos - float64(particle_cell_x)
            dx_right = float64(particle_cell_x) + 1.0 - x_pos
            dy_down = y_pos - float64(particle_cell_y)
            dy_up = float64(particle_cell_y) + 1.0 - y_pos

            overlaps_left = dx_left < kernel_width
            overlaps_right = dx_right < kernel_width
            overlaps_down = dy_down < kernel_width
            overlaps_up = dy_up < kernel_width

            if not (overlaps_left or overlaps_right or overlaps_down or overlaps_up):
                # Very simple case - no overlaps.
                image[particle_cell_x, particle_cell_y] += mass * inverse_cell_area
            else:
                # Use pre-calculated kernel with a basic dither to lay down
                # overlap
                for x_dither_cell in range(0, 2 * DITHER_EVALUATIONS):
                    float_x_dither_cell = float64(x_dither_cell)
                    pixel_x = int32(
                        float_res
                        * (
                            x_pos
                            + (float_x_dither_cell * float_DITHER_EVALUATIONS_inv - 1.0)
                            * kernel_width
                        )
                    )
                    for y_dither_cell in range(0, 2 * DITHER_EVALUATIONS):
                        float_y_dither_cell = float64(y_dither_cell)
                        pixel_y = int32(
                            float_res
                            * (
                                y_pos
                                + (
                                    float_y_dither_cell * float_DITHER_EVALUATIONS_inv
                                    - 1.0
                                )
                                * kernel_width
                            )
                        )

                        image[pixel_x, pixel_y] += (
                            float_mass * dithered_kernel[x_dither_cell, y_dither_cell]
                        )

        else:
            # The number of times each pixel is subsampled.
            subsample_factor = max(
                1, 2 * int32(ceil(float_MIN_KERNEL_EVALUATIONS / float_cells_spanned))
            )
            float_subsample_factor = float64(subsample_factor)
            inv_float_subsample_factor = 1.0 / float_subsample_factor
            inv_float_subsample_factor_square = (
                inv_float_subsample_factor * inv_float_subsample_factor
            )

            # Now we loop over the square of cells that the kernel lives in
            for cell_x in range(
                # Ensure that the lowest x value is 0, otherwise we'll segfault
                max(0, particle_cell_x - cells_spanned),
                # Ensure that the highest x value lies within the array bounds,
                # otherwise we'll segfault (oops).
                min(particle_cell_x + cells_spanned + 1, maximal_array_index),
            ):
                float_cell_x = float64(cell_x)
                for cell_y in range(
                    max(0, particle_cell_y - cells_spanned),
                    min(particle_cell_y + cells_spanned + 1, maximal_array_index),
                ):
                    float_cell_y = float64(cell_y)
                    # Now we subsample the pixels to get a more accurate determination
                    # of the kernel weight. We take the mean of the kernel evaluations
                    # within a given pixel and apply this as the true 'kernel evaluation'.
                    kernel_eval = float64(0.0)

                    for subsample_x in range(0, subsample_factor):
                        subsample_position_x = (
                            float64(subsample_x) + 0.5
                        ) * inv_float_subsample_factor

                        distance_x = (
                            float_cell_x + subsample_position_x
                        ) * pixel_width - x_pos

                        distance_x_2 = distance_x * distance_x

                        for subsample_y in range(0, subsample_factor):
                            subsample_position_y = (
                                float64(subsample_y) + 0.5
                            ) * inv_float_subsample_factor

                            distance_y = (
                                float_cell_y + subsample_position_y
                            ) * pixel_width - y_pos

                            distance_y_2 = distance_y * distance_y

                            r = sqrt(distance_x_2 + distance_y_2)
                            kernel_eval += kernel(r, kernel_width)

                    image[cell_x, cell_y] += (
                        float_mass * kernel_eval * inv_float_subsample_factor_square
                    )

    return image


@jit(nopython=True, fastmath=True, parallel=True)
def scatter_parallel(
    x: float64, y: float64, m: float32, h: float32, res: int
) -> ndarray:
    """
    Parallel implementation of scatter

    Creates a weighted scatter plot. Computes contributions from
    particles with positions (`x`,`y`) with smoothing lengths `h`
    weighted by quantities `m`.
    This ignores boundary effects.

    Parameters
    ----------
    x : np.array[float64]
        array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.array[float64]
        array of y-positions of the particles. Must be bounded by [0, 1].

    m : np.array[float32]
        array of masses (or otherwise weights) of the particles

    h : np.array[float32]
        array of smoothing lengths of the particles

    res : int
        the number of pixels along one axis, i.e. this returns a square
        of res * res.

    Returns
    -------

    np.array[float32, float32, float32]
        pixel grid of quantity

    See Also
    --------

    scatter : Creates 2D scatter plot from SWIFT data

    Notes
    -----

    Explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.

    Uses 4x the number of sampling points as in scatter_parallel in subsampled.py
    """
    # Same as scatter, but executes in parallel! This is actually trivial,
    # we just make NUM_THREADS images and add them together at the end.

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
