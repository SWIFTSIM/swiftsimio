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
    r"""
    Creates a weighted voxel grid

    Computes contributions to a voxel grid from particles with positions
    (`x`,`y`,`z`) with smoothing lengths `h` weighted by quantities `m`.
    This ignores boundary effects.
    Parameters
    ----------
    x : np.array[float64]
        array of x-positions of the particles. Must be bounded by [0, 1].
    y : np.array[float64]
        array of y-positions of the particles. Must be bounded by [0, 1].
    z : np.array[float64]
        array of z-positions of the particles. Must be bounded by [0, 1].
    m : np.array[float32]
        array of masses (or otherwise weights) of the particles
    h : np.array[float32]
        array of smoothing lengths of the particles
    res : int
        the number of voxels along one axis, i.e. this returns a cube
         of res * res * res..

    Returns
    -------
    np.array[float32, float32, float32]
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
    r"""
    Parallel implementation of scatter

    Compute contributions to a voxel grid from particles with positions
    (`x`,`y`,`z`) with smoothing lengths `h` weighted by quantities `m`.
    This ignores boundary effects.
    Parameters
    ----------
    x : array of float64 
        array of x-positions of the particles. Must be bounded by [0, 1].
    y : array of float64 
        array of y-positions of the particles. Must be bounded by [0, 1].
    z : array of float64 
        array of z-positions of the particles. Must be bounded by [0, 1].
    m : array of float32 
        array of masses (or otherwise weights) of the particles
    h : array of float32 
        array of smoothing lengths of the particles
    res : int
        the number of voxels along one axis, i.e. this returns a cube
         of res * res * res..

    Returns
    -------
    ndarray of float32
        voxel grid of quantity 
    
    See Also
    --------
    scatter : Create voxel grid of quantity
    slice_scatter : Create scatter plot of a slice of data
    slice_scatter_parallel : Create scatter plot of a slice of data in parallel

    Notes
    -----
    ALEXEI: Check with Josh if this note still holds
    Explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.
    """
    #Same as scatter, but executes in parallel! This is actually trivial,
    #we just make NUM_THREADS images and add them together at the end.

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
    Creates a 3D render of a SWIFT dataset, weighted by data field, in the
    form of a voxel grid.

    Parameters
    ----------
    data : SWIFTDataset
        Dataset from which slice is extracted

    resolution : int
        Specifies size of return array
        
    project : str, optional
        Data field to be projected. Default is mass. If None then simply
        count number of particles (ALEXEI: check wording with Josh)
    
    parallel : bool
        used to determine if we will create the image in parallel. This 
        defaults to False, but can speed up the creation of large images 
        significantly at the cost of increased memory usage.

    region : array, optional
        determines where the image will be created
        (this corresponds to the left and right-hand edges, and top and bottom
        edges, and front and back edges) if it is not None. It should have a
        length of six, and take the form:

        [x_min, x_max, y_min, y_max, z_min, z_max]

        Particles outside of this range are still considered if their
        smoothing lengths overlap with the range.

    Returns
    -------
    ndarray of float32
        Creates a `resolution` x `resolution` x `resolution` array and 
        returns it, without appropriate units.

    See Also
    --------
    slice_gas_pixel_grid : Creates a 2D slice of a SWIFT dataset

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
        x_min = (0 * box_x).to(box_x.units)
        x_max = box_x
        y_min = (0 * box_y).to(box_y.units)
        y_max = box_y
        z_min = (0 * box_z).to(box_z.units)
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
    Creates a 3D voxel grid of a SWIFT dataset, weighted by data field

    Parameters
    ----------
    data : SWIFTDataset
        Dataset from which slice is extracted

    resolution : int
        Specifies size of return array
        
    project : str, optional
        Data field to be projected. Default is mass. If None then simply
        count number of particles (ALEXEI: check wording with Josh)
    
    parallel : bool
        used to determine if we will create the image in parallel. This 
        defaults to False, but can speed up the creation of large images 
        significantly at the cost of increased memory usage.

    region : array, optional
        determines where the image will be created
        (this corresponds to the left and right-hand edges, and top and bottom
        edges, and front and back edges) if it is not None. It should have a
        length of six, and take the form:

        [x_min, x_max, y_min, y_max, z_min, z_max]

        Particles outside of this range are still considered if their
        smoothing lengths overlap with the range.

    Returns
    -------
    ndarray of float32
        a `resolution` x `resolution` x `resolution` array of the contribution
        of the projected data field to the voxel grid from all of the particles

    See Also
    --------
    slice_gas : Creates a 2D slice of a SWIFT dataset with appropriate units
    render_gas_voxel_grid : Creates a 3D voxel grid of a SWIFT dataset

    Notes
    -----
    This is a wrapper function for slice_gas_pixel_grid ensuring that output units are
    appropriate
    """

    image = render_gas_voxel_grid(data, resolution, project, parallel, region=region)

    if region is not None:
        x_range = region[1] - region[0]
        y_range = region[3] - region[2]
        z_range = region[5] - region[4]
        units = 1.0 / (x_range * y_range * z_range)
        units.convert_to_units(1.0 / (x_range.units * y_range.units * z_range.units))
    else:
        units = 1.0 / (
            data.metadata.boxsize[0]
            * data.metadata.boxsize[1]
            * data.metadata.boxsize[2]
        )
        units.convert_to_units(1.0 / data.metadata.boxsize.units ** 3)

    if project is not None:
        units *= getattr(data.gas, project).units

    return image * units
