from numba import cuda
from math import sqrt, ceil
from numpy import float64, float32, int32, ndarray
from swiftsimio.optional_packages import CUDA_AVAILABLE

kernel_gamma = float32(1.897367)


@cuda.jit("float32(float32, float32)", device=True)
def kernel(r: float32, H: float32):
    """
    Single precision kernel implementation for swiftsimio.

    This is the Wendland-C2 kernel as shown in Denhen & Aly (2012) [1]_.

    Parameters
    ----------

    r : float32
        radius used in kernel computation

    H : float32
        kernel width (i.e. radius of compact support for the kernel)

    Returns
    -------

    float32
        Contribution to the density by the particle

    See Also
    --------

    kernel_double_precision

    References
    ----------

    .. [1] Dehnen W., Aly H., 2012, MNRAS, 425, 1068
    """
    kernel_constant = float32(2.22817109)

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


@cuda.jit
def scatter_gpu(x: float64, y: float64, m: float32, h: float32, img: float32):
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

    img : np.array[float32]
        The output image.

    Notes
    -----

    Explicitly defining the types in this function allows
    for a performance improvement.
    """
    # Output array for our image
    res = img.shape[0]
    maximal_array_index = int32(res)

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    float_res = float32(res)
    pixel_width = 1.0 / float_res

    # We need this for combining with the x_pos and y_pos variables.
    float_res_64 = float64(res)

    # Pre-calculate this constant for use with the above
    inverse_cell_area = res * res

    i = cuda.grid(1)
    if i < len(x):
        # Get the correct particle
        x_pos = x[i]
        y_pos = y[i]
        mass = m[i]
        hsml = h[i]

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
            return

        if cells_spanned <= 1:
            # Easygame, gg
            cuda.atomic.add(
                img, (particle_cell_x, particle_cell_y), mass * inverse_cell_area
            )
        else:
            # Now we loop over the square of cells that the kernel lives in
            for cell_x in range(
                # Ensure that the lowest x value is 0, otherwise we'll segfault
                max(0, particle_cell_x - cells_spanned),
                # Ensure that the highest x value lies within the array bounds,
                # otherwise we'll segfault (oops).
                min(particle_cell_x + cells_spanned + 1, maximal_array_index),
            ):
                # The distance in x to our new favourite cell
                # remember that our x, y are all in a box of [0, 1]
                # calculate the distance to the cell center
                distance_x = (float32(cell_x) + 0.5) * pixel_width
                distance_x -= float32(x_pos)
                distance_x_2 = distance_x * distance_x
                for cell_y in range(
                    max(0, particle_cell_y - cells_spanned),
                    min(particle_cell_y + cells_spanned + 1, maximal_array_index),
                ):
                    distance_y = (float32(cell_y) + 0.5) * pixel_width
                    distance_y -= float32(y_pos)
                    distance_y_2 = distance_y * distance_y

                    r = sqrt(distance_x_2 + distance_y_2)

                    kernel_eval = kernel(r, kernel_width)

                    cuda.atomic.add(img, (cell_x, cell_y), mass * kernel_eval)


def scatter(x: float64, y: float64, m: float32, h: float32, res: int) -> ndarray:
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
    a performance improvement.
    """
    if not CUDA_AVAILABLE:
        raise Exception(
            "Unable to load the GPU function. "
            "Please check your module numba.cuda."
        )

    output = cuda.device_array((res, res), dtype=float32)
    output[:] = 0

    n_part = len(x)
    threads_per_block = 16
    blocks_per_grid = ceil(n_part / threads_per_block)
    scatter[blocks_per_grid, threads_per_block](x, y, m, h, output)

    return output.copy_to_host()


scatter_parallel = scatter
