"""
Ray tracing module for visualisation.
"""

# There should be three implementations here:
# - An example case that uses 'screening' with multiple panes.
# - An example case that builds a 3D mesh and uses that
# - An example case that uses a 'real' algorithm (even if it has to
#   be single-threaded)

import numpy as np
import math

from swiftsimio.visualisation.projection_backends.kernels import (
    kernel_gamma,
    kernel_double_precision as kernel,
)

from swiftsimio.accelerated import jit, prange, NUM_THREADS


@jit(nopython=True, fastmath=True)
def core_panels(
    x: np.float64,
    y: np.float64,
    z: np.float64,
    h: np.float32,
    m: np.float32,
    res: int,
    panels: int,
    min_z: np.float64,
    max_z: np.float64,
) -> np.array:
    """
    Creates a 2D array of the projected density of particles in a 3D volume using the
    'renormalised' strategy, with multiple panels across the z-range.

    Parameters
    ----------

    x: np.array[np.float64]
        The x-coordinates of the particles.

    y: np.array[np.float64]
        The y-coordinates of the particles.

    z: np.array[np.float64]
        The z-coordinates of the particles.

    h: np.array[np.float32]
        The smoothing lengths of the particles.

    m: np.array[np.float32]
        The masses of the particles.

    res: int
        The resolution of the output array.

    panels: int
        The number of panels to use in the z-direction.

    min_z: np.float64
        The minimum z-coordinate of the volume.

    max_z: np.float64
        The maximum z-coordinate of the volume.

    Returns
    -------

    A 3D array of shape (res, res, panels) containing the projected density in each pixel.
    """
    output = np.zeros((res, res, panels))
    maximal_array_index = res - 1

    number_of_particles = len(x)
    float_res = float(res)
    pixel_width = 1.0 / float_res

    assert len(y) == number_of_particles
    assert len(z) == number_of_particles
    assert len(h) == number_of_particles
    assert len(m) == number_of_particles

    z_per_panel = (max_z - min_z) / panels

    inverse_cell_area = float_res * float_res

    for i in range(number_of_particles):
        panel = int(z[i] / z_per_panel)

        if panel < 0 or panel >= panels:
            continue

        particle_cell_x = int(float_res * x[i])
        particle_cell_y = int(float_res * y[i])

        kernel_width = kernel_gamma * h[i]
        cells_spanned = int(1.0 + kernel_width * float_res)

        if (
            particle_cell_x + cells_spanned < 0
            or particle_cell_x - cells_spanned > maximal_array_index
            or particle_cell_y + cells_spanned < 0
            or particle_cell_y - cells_spanned > maximal_array_index
        ):
            # Can happily skip this particle
            continue

        if cells_spanned <= 1:
            if (
                particle_cell_x >= 0
                and particle_cell_x <= maximal_array_index
                and particle_cell_y >= 0
                and particle_cell_y <= maximal_array_index
            ):
                output[particle_cell_x, particle_cell_y, panel] += (
                    m[i] * inverse_cell_area
                )
            continue

        normalisation = 0.0

        for cell_x in range(
            particle_cell_x - cells_spanned, particle_cell_x + cells_spanned + 1
        ):
            distance_x = (float(cell_x) + 0.5) * pixel_width - x[i]
            distance_x_2 = distance_x * distance_x

            for cell_y in range(
                particle_cell_y - cells_spanned, particle_cell_y + cells_spanned + 1
            ):
                distance_y = (float(cell_y) + 0.5) * pixel_width - y[i]
                distance_y_2 = distance_y * distance_y

                r = math.sqrt(distance_x_2 + distance_y_2)

                normalisation += kernel(r, kernel_width)

        # Now have the normalisation
        normalisation = m[i] * inverse_cell_area / normalisation

        for cell_x in range(
            # Ensure that the lowest x value is 0, otherwise we'll segfault
            max(0, particle_cell_x - cells_spanned),
            # Ensure that the highest x value lies within the array bounds,
            # otherwise we'll segfault (oops).
            min(particle_cell_x + cells_spanned + 1, maximal_array_index + 1),
        ):
            distance_x = (float(cell_x) + 0.5) * pixel_width - x[i]
            distance_x_2 = distance_x * distance_x

            for cell_y in range(
                max(0, particle_cell_y - cells_spanned),
                min(particle_cell_y + cells_spanned + 1, maximal_array_index + 1),
            ):
                distance_y = (float(cell_y) + 0.5) * pixel_width - y[i]
                distance_y_2 = distance_y * distance_y

                r = math.sqrt(distance_x_2 + distance_y_2)

                output[cell_x, cell_y, panel] += kernel(r, kernel_width) * normalisation

    return output


@jit(nopython=True, fastmath=True)
def core_panels_parallel(
    x: np.float64,
    y: np.float64,
    z: np.float64,
    h: np.float32,
    m: np.float32,
    res: int,
    panels: int,
    min_z: np.float64,
    max_z: np.float64,
):
    # Same as scatter, but executes in parallel! This is actually trivial,
    # we just make NUM_THREADS images and add them together at the end.

    number_of_particles = x.size
    core_particles = number_of_particles // NUM_THREADS

    output = np.zeros((res, res, panels), dtype=np.float32)

    for thread in prange(NUM_THREADS):
        # Left edge is easy, just start at 0 and go to 'final'
        left_edge = thread * core_particles

        # Right edge is harder in case of left over particles...
        right_edge = thread + 1

        if right_edge == NUM_THREADS:
            right_edge = number_of_particles
        else:
            right_edge *= core_particles

        output += core_panels(
            x[left_edge:right_edge],
            y[left_edge:right_edge],
            z[left_edge:right_edge],
            h[left_edge:right_edge],
            m[left_edge:right_edge],
            res,
            panels,
            min_z,
            max_z,
        )

    return output


# --- Functions that actually perform the 'ray tracing'.


def transfer_function(value, width, center):
    """
    A simple gaussian transfer function centered around a specific value.
    """
    return (
        1
        / (width * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((value - center) / width) ** 2)
    )


@jit(fastmath=True, nopython=True)
def integrate_ray_numba_specific(
    input: np.array, red: float, green: float, blue: float, center: float, width: float
):
    """
    Given a ray, integrate the transfer function along it
    """

    value = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    color = np.array([red, green, blue], dtype=np.float32)

    for i in input:
        value += (
            color
            * 1
            / (width * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((i - center) / width) ** 2)
        )

    return value / len(input)


@jit(fastmath=True, nopython=True)
def integrate_ray_numba_nocolor(input: np.array, center: float, width: float):
    """
    Given a ray, integrate the transfer function along it
    """

    value = np.float32(0.0)

    for i in input:
        value *= 0.99
        value += (
            1
            / (width * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((i - center) / width) ** 2)
        )

    return np.float32(value / len(input))


# #%%
# data = np.load("voxel_1024.npy")
# # %%
# log_data = np.log10(data)
# # %%
# transfer = lambda x: transfer_function(x, np.mean(log_data), np.std(log_data) * 0.5)
# # %%
# color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
# #%%
# from tqdm import tqdm
# # %%
# @numba.njit(fastmath=True)
# def make_grid(color, center, width):
#     output = np.zeros((len(log_data), len(log_data[0])), dtype=np.float32)
#     for x in numba.prange(len(log_data)):
#         for y in range(len(log_data)):
#             data = log_data[x, y]

#             value = np.float32(0.0)

#             for index, i in enumerate(data):
#                 factor = index / len(data)

#                 if factor > 0.5:
#                     factor = 1.0 - factor

#                 value += (
#                     1
#                     / (width * np.sqrt(2.0 * np.pi))
#                     * np.exp(-0.5 * ((i - center) / width) ** 2)
#                 )

#             output[x, y] = value

#     return output


# # %%
# import matplotlib.pyplot as plt
# import swiftascmaps
# # %%
# std = np.std(log_data)
# width = 0.05
# centers = [np.mean(log_data) + x * std for x in [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]]
# # %%
# colors = plt.get_cmap("swift.nineteen_eighty_nine")((np.linspace(0, 1, len(centers))))[
#     :, :3
# ]

# grids = [
#     make_grid(color, center, width) for color, center in zip(colors, centers)
# ]
# #%%

# #%%
# make_image = lambda x, y: np.array([x * y[0], x * y[1], x * y[2]]).T
# images = [make_image(grid / np.max(grid), color) for color, grid in zip(colors, grids)]
# # %%
# combined_image = sum(images)
# plt.imsave("test.png", combined_image / np.max(combined_image))
# # %%
# for id, image in zip(centers, images):
#     plt.imsave(f"test{id}.png", image)
# # %%
