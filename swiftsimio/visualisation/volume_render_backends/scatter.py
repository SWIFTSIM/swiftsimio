"""
Basic volume render for SPH data.

Takes the 3D positions of the particles and projects them onto a grid.
"""

from math import sqrt, ceil
import numpy as np

from swiftsimio.accelerated import jit, NUM_THREADS, prange

from swiftsimio.visualisation.slice_backends.sph import kernel, kernel_gamma


@jit(nopython=True, fastmath=True)
def scatter(
    x: np.float64,
    y: np.float64,
    z: np.float64,
    m: np.float32,
    h: np.float32,
    res: int,
    box_x: np.float64 = 0.0,
    box_y: np.float64 = 0.0,
    box_z: np.float64 = 0.0,
) -> np.ndarray:
    """
    Create a weighted voxel grid.

    Computes contributions to a voxel grid from particles with positions
    (`x`,`y`,`z`) with smoothing lengths `h` weighted by quantities `m`.
    This includes periodic boundary effects.

    Parameters
    ----------
    x : np.np.array[np.float64]
        np.array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.np.array[np.float64]
        np.array of y-positions of the particles. Must be bounded by [0, 1].

    z : np.np.array[np.float64]
        np.array of z-positions of the particles. Must be bounded by [0, 1].

    m : np.np.array[np.float32]
        np.array of masses (or otherwise weights) of the particles

    h : np.np.array[np.float32]
        np.array of smoothing lengths of the particles

    res : int
        the number of voxels along one axis, i.e. this returns a cube
        of res * res * res.

    box_x: np.float64
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.

    box_y: np.float64
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.

    box_z: np.float64
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping

    Returns
    -------
    np.np.array[np.float32, np.float32, np.float32]
        voxel grid of quantity

    See Also
    --------
    scatter_parallel : Parallel implementation of this function
    slice_scatter : Create scatter plot of a slice of data
    slice_scatter_parallel : Create scatter plot of a slice of data in parallel

    Notes
    -----
    Explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba np.ones.
    """
    # Output np.array for our image
    image = np.zeros((res, res, res), dtype=np.float32)
    maximal_array_index = np.int32(res) - 1

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    float_res = np.float32(res)
    pixel_width = 1.0 / float_res

    # We need this for combining with the x_pos and y_pos variables.
    float_res_64 = np.float64(res)

    # If the kernel width is smaller than this, we drop to just PIC method
    drop_to_single_cell = pixel_width * 0.5

    # Pre-calculate this constant for use with the above
    inverse_cell_volume = float_res * float_res * float_res

    if box_x == 0.0:
        xshift_min = 0
        xshift_max = 1
    else:
        xshift_min = -1  # x_min is always at x=0
        xshift_max = ceil(1 / box_x) + 1  # tile the box to cover [0, 1]
    if box_y == 0.0:
        yshift_min = 0
        yshift_max = 1
    else:
        yshift_min = -1  # y_min is always at y=0
        yshift_max = ceil(1 / box_y) + 1  # tile the box to cover [0, 1]
    if box_z == 0.0:
        zshift_min = 0
        zshift_max = 1
    else:
        zshift_min = -1  # z_min is always at z=0
        zshift_max = ceil(1 / box_z) + 1  # tile the box to cover [0, 1]

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

                    # Calculate the cell that this particle; use the 64 bit version of the
                    # resolution as this is the same type as the positions
                    particle_cell_x = np.int32(float_res_64 * x_pos)
                    particle_cell_y = np.int32(float_res_64 * y_pos)
                    particle_cell_z = np.int32(float_res_64 * z_pos)

                    # SWIFT stores hsml as the FWHM.
                    kernel_width = kernel_gamma * hsml

                    # The number of cells that this kernel spans
                    cells_spanned = np.int32(1.0 + kernel_width * float_res)

                    if (
                        particle_cell_x + cells_spanned < 0
                        or particle_cell_x - cells_spanned > maximal_array_index
                        or particle_cell_y + cells_spanned < 0
                        or particle_cell_y - cells_spanned > maximal_array_index
                        or particle_cell_z + cells_spanned < 0
                        or particle_cell_z - cells_spanned > maximal_array_index
                    ):
                        # Can happily skip this particle
                        continue

                    if kernel_width < drop_to_single_cell:
                        # Easygame, gg
                        if (
                            particle_cell_x >= 0
                            and particle_cell_x <= maximal_array_index
                            and particle_cell_y >= 0
                            and particle_cell_y <= maximal_array_index
                            and particle_cell_z >= 0
                            and particle_cell_z <= maximal_array_index
                        ):
                            image[
                                particle_cell_x, particle_cell_y, particle_cell_z
                            ] += mass * inverse_cell_volume
                    else:
                        # Now we loop over the square of cells that the kernel lives in
                        for cell_x in range(
                            # Ensure that the lowest x value is 0, otherwise we segfault
                            max(0, particle_cell_x - cells_spanned),
                            # Ensure that the highest x value lies within the np.array
                            # bounds, otherwise we'll segfault (oops).
                            min(
                                particle_cell_x + cells_spanned, maximal_array_index + 1
                            ),
                        ):
                            # The distance in x to our new favourite cell -- remember that
                            # our x, y are all in a box of [0, 1]; calculate the distance
                            # to the cell centre
                            distance_x = (
                                np.float32(cell_x) + 0.5
                            ) * pixel_width - np.float32(x_pos)
                            distance_x_2 = distance_x * distance_x
                            for cell_y in range(
                                max(0, particle_cell_y - cells_spanned),
                                min(
                                    particle_cell_y + cells_spanned,
                                    maximal_array_index + 1,
                                ),
                            ):
                                distance_y = (
                                    np.float32(cell_y) + 0.5
                                ) * pixel_width - np.float32(y_pos)
                                distance_y_2 = distance_y * distance_y
                                for cell_z in range(
                                    max(0, particle_cell_z - cells_spanned),
                                    min(
                                        particle_cell_z + cells_spanned,
                                        maximal_array_index + 1,
                                    ),
                                ):
                                    distance_z = (
                                        np.float32(cell_z) + 0.5
                                    ) * pixel_width - np.float32(z_pos)
                                    distance_z_2 = distance_z * distance_z

                                    r = sqrt(distance_x_2 + distance_y_2 + distance_z_2)

                                    kernel_eval = kernel(r, kernel_width)

                                    image[cell_x, cell_y, cell_z] += mass * kernel_eval

    return image


@jit(nopython=True, fastmath=True)
def scatter_limited_z(
    x: np.float64,
    y: np.float64,
    z: np.float64,
    m: np.float32,
    h: np.float32,
    res: int,
    res_ratio_z: int,
    box_x: np.float64 = 0.0,
    box_y: np.float64 = 0.0,
    box_z: np.float64 = 0.0,
) -> np.ndarray:
    """
    Create a weighted voxel grid.

    Computes contributions to a voxel grid from particles with positions
    (`x`,`y`,`z`) with smoothing lengths `h` weighted by quantities `m`.
    This includes periodic boundary effects.

    Parameters
    ----------
    x : np.np.array[np.float64]
        np.array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.np.array[np.float64]
        np.array of y-positions of the particles. Must be bounded by [0, 1].

    z : np.np.array[np.float64]
        np.array of z-positions of the particles. Must be bounded by [0, 1].

    m : np.np.array[np.float32]
        np.array of masses (or otherwise weights) of the particles

    h : np.np.array[np.float32]
        np.array of smoothing lengths of the particles

    res : int
        the number of voxels along one axis, i.e. this returns a cube
        of res * res * res.

    res_ratio_z: int
        the number of voxels along the x and y axes relative to the z
        axis. If this is, for instance, 8, and the res is 128, then the
        output np.array will be 128 x 128 x 16.

    box_x: np.float64
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.

    box_y: np.float64
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.

    box_z: np.float64
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping

    Returns
    -------
    np.np.array[np.float32, np.float32, np.float32]
        voxel grid of quantity

    See Also
    --------
    scatter_parallel : Parallel implementation of this function
    slice_scatter : Create scatter plot of a slice of data
    slice_scatter_parallel : Create scatter plot of a slice of data in parallel

    Notes
    -----
    Explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba np.ones.
    """
    # Output np.array for our image
    res_z = res // res_ratio_z
    image = np.zeros((res, res, res_z), dtype=np.float32)
    maximal_array_index = np.int32(res) - 1
    maximal_array_index_z = np.int32(res_z) - 1

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    float_res = np.float32(res)
    float_res_z = np.float32(res_z)
    pixel_width = 1.0 / float_res
    pixel_width_z = 1.0 / float_res_z

    # We need this for combining with the x_pos and y_pos variables.
    float_res_64 = np.float64(res)
    float_res_z_64 = np.float64(res_z)

    # If the kernel width is smaller than this, we drop to just PIC method
    drop_to_single_cell = pixel_width * 0.5
    drop_to_single_cell_z = pixel_width_z * 0.5

    # Pre-calculate this constant for use with the above
    inverse_cell_volume = float_res * float_res * float_res_z

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

                    # Calculate the cell that this particle; use the 64 bit version of the
                    # resolution as this is the same type as the positions
                    particle_cell_x = np.int32(float_res_64 * x_pos)
                    particle_cell_y = np.int32(float_res_64 * y_pos)
                    particle_cell_z = np.int32(float_res_z_64 * z_pos)

                    # SWIFT stores hsml as the FWHM.
                    kernel_width = kernel_gamma * hsml

                    # The number of cells that this kernel spans
                    cells_spanned = np.int32(1.0 + kernel_width * float_res)
                    cells_spanned_z = np.int32(1.0 + kernel_width * float_res_z)

                    if (
                        particle_cell_x + cells_spanned < 0
                        or particle_cell_x - cells_spanned > maximal_array_index
                        or particle_cell_y + cells_spanned < 0
                        or particle_cell_y - cells_spanned > maximal_array_index
                        or particle_cell_z + cells_spanned_z < 0
                        or particle_cell_z - cells_spanned_z > maximal_array_index_z
                    ):
                        # Can happily skip this particle
                        continue

                    if (
                        kernel_width < drop_to_single_cell
                        or kernel_width < drop_to_single_cell_z
                    ):
                        # Easygame, gg
                        if (
                            particle_cell_x >= 0
                            and particle_cell_x <= maximal_array_index
                            and particle_cell_y >= 0
                            and particle_cell_y <= maximal_array_index
                            and particle_cell_z >= 0
                            and particle_cell_z <= maximal_array_index_z
                        ):
                            image[
                                particle_cell_x, particle_cell_y, particle_cell_z
                            ] += mass * inverse_cell_volume
                    else:
                        # Now we loop over the square of cells that the kernel lives in
                        for cell_x in range(
                            # Ensure that the lowest x value is 0, otherwise we segfault
                            max(0, particle_cell_x - cells_spanned),
                            # Ensure that the highest x value lies within the np.array
                            # bounds, otherwise we'll segfault (oops).
                            min(
                                particle_cell_x + cells_spanned, maximal_array_index + 1
                            ),
                        ):
                            # The distance in x to our new favourite cell -- remember that
                            # our x, y are all in a box of [0, 1]; calculate the distance
                            # to the cell centre
                            distance_x = (
                                np.float32(cell_x) + 0.5
                            ) * pixel_width - np.float32(x_pos)
                            distance_x_2 = distance_x * distance_x
                            for cell_y in range(
                                max(0, particle_cell_y - cells_spanned),
                                min(
                                    particle_cell_y + cells_spanned,
                                    maximal_array_index + 1,
                                ),
                            ):
                                distance_y = (
                                    np.float32(cell_y) + 0.5
                                ) * pixel_width - np.float32(y_pos)
                                distance_y_2 = distance_y * distance_y
                                for cell_z in range(
                                    max(0, particle_cell_z - cells_spanned_z),
                                    min(
                                        particle_cell_z + cells_spanned_z,
                                        maximal_array_index_z + 1,
                                    ),
                                ):
                                    distance_z = (
                                        np.float32(cell_z) + 0.5
                                    ) * pixel_width_z - np.float32(z_pos)
                                    distance_z_2 = distance_z * distance_z

                                    r = sqrt(distance_x_2 + distance_y_2 + distance_z_2)

                                    kernel_eval = kernel(r, kernel_width)

                                    image[cell_x, cell_y, cell_z] += mass * kernel_eval

    return image


@jit(nopython=True, fastmath=True, parallel=True)
def scatter_parallel(
    x: np.float64,
    y: np.float64,
    z: np.float64,
    m: np.float32,
    h: np.float32,
    res: int,
    res_ratio_z: int = 1,
    box_x: np.float64 = 0.0,
    box_y: np.float64 = 0.0,
    box_z: np.float64 = 0.0,
) -> np.ndarray:
    """
    Parallel implementation of scatter.

    Compute contributions to a voxel grid from particles with positions
    (`x`,`y`,`z`) with smoothing lengths `h` weighted by quantities `m`.
    This ignores boundary effects.

    Parameters
    ----------
    x : np.array of np.float64
        np.array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.array of np.float64
        np.array of y-positions of the particles. Must be bounded by [0, 1].

    z : np.array of np.float64
        np.array of z-positions of the particles. Must be bounded by [0, 1].

    m : np.array of np.float32
        np.array of masses (or otherwise weights) of the particles

    h : np.array of np.float32
        np.array of smoothing lengths of the particles

    res : int
        the number of voxels along one axis, i.e. this returns a cube
        of res * res * res.

    res_ratio_z: int
        the number of voxels along the x and y axes relative to the z

    box_x: np.float64
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.

    box_y: np.float64
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping.

    box_z: np.float64
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping

    Returns
    -------
    np.ndarray of np.float32
        voxel grid of quantity

    See Also
    --------
    scatter : Create voxel grid of quantity
    slice_scatter : Create scatter plot of a slice of data
    slice_scatter_parallel : Create scatter plot of a slice of data in parallel

    Notes
    -----
    Explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba np.ones.

    """
    # Same as scatter, but executes in parallel! This is actually trivial,
    # we just make NUM_THREADS images and add them together at the end.

    number_of_particles = x.size
    core_particles = number_of_particles // NUM_THREADS

    output = np.zeros((res, res, res), dtype=np.float32)

    for thread in prange(NUM_THREADS):
        # Left edge is easy, just start at 0 and go to 'final'
        left_edge = thread * core_particles

        # Right edge is harder in case of left over particles...
        right_edge = thread + 1

        if right_edge == NUM_THREADS:
            right_edge = number_of_particles
        else:
            right_edge *= core_particles

        # using kwargs is unsupported in numba
        if res_ratio_z == 1:
            output += scatter(
                x=x[left_edge:right_edge],
                y=y[left_edge:right_edge],
                z=z[left_edge:right_edge],
                m=m[left_edge:right_edge],
                h=h[left_edge:right_edge],
                res=res,
                box_x=box_x,
                box_y=box_y,
                box_z=box_z,
            )
        else:
            output += scatter_limited_z(
                x=x[left_edge:right_edge],
                y=y[left_edge:right_edge],
                z=z[left_edge:right_edge],
                m=m[left_edge:right_edge],
                h=h[left_edge:right_edge],
                res=res,
                res_ratio_z=res_ratio_z,
                box_x=box_x,
                box_y=box_y,
                box_z=box_z,
            )

    return output
