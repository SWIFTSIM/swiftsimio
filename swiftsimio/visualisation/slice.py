"""
Sub-module for slice plots in SWFITSIMio.
"""

from typing import Union, Optional
from numpy import float32, array, ones, matmul
from unyt import unyt_array, unyt_quantity
from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.visualisation.slice_backends import (
    backends,
    backends_parallel,
    backends_get_hsml,
)

from swiftsimio.visualisation.slice_backends.sph import (
    kernel,
    kernel_constant,
    kernel_gamma,
)

slice_scatter = backends["sph"]
slice_scatter_parallel = backends_parallel["sph"]


def slice_gas_pixel_grid(
    data: SWIFTDataset,
    resolution: int,
    z_slice: Optional[unyt_quantity] = None,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    rotation_matrix: Union[None, array] = None,
    rotation_center: Union[None, unyt_array] = None,
    region: Union[None, unyt_array] = None,
    backend: str = "sph",
    periodic: bool = True,
):
    """
    Creates a 2D slice of a SWIFT dataset, weighted by data field, in the
    form of a pixel grid.

    Parameters
    ----------
    data : SWIFTDataset
        Dataset from which slice is extracted

    resolution : int
        Specifies size of return array

    z_slice : unyt_quantity
        Specifies the location along the z-axis where the slice is to be
        extracted, relative to the rotation center or the origin of the box
        if no rotation center is provided. If the perspective is rotated
        this value refers to the location along the rotated z-axis.

    project : str, optional
        Data field to be projected. Default is mass. If None then simply
        count number of particles

    parallel : bool
        used to determine if we will create the image in parallel. This
        defaults to False, but can speed up the creation of large images
        significantly at the cost of increased memory usage.

    rotation_matrix: np.array, optional
        Rotation matrix (3x3) that describes the rotation of the box around
        ``rotation_center``. In the default case, this provides a slice
        perpendicular to the z axis.

    rotation_center: np.array, optional
        Center of the rotation. If you are trying to rotate around a galaxy, this
        should be the most bound particle.

    region : unyt_array, optional
        determines where the image will be created
        (this corresponds to the left and right-hand edges, and top and bottom edges)
        if it is not None. It should have a length of four, and take the form:

        [x_min, x_max, y_min, y_max]

        Particles outside of this range are still considered if their
        smoothing lengths overlap with the range.

    backend : str, optional
        Backend to use. Choices are "sph" (default) for interpolation using kernel
        weights or "nearest_neighbours" for nearest neighbour interpolation.

    periodic : bool, optional
        Account for periodic boundaries for the simulation box?
        Default is ``True``.

    Returns
    -------
    ndarray of float32
        Creates a `resolution` x `resolution` array and returns it,
        without appropriate units.

    See Also
    --------
    render_gas_voxel_grid : Creates a 3D voxel grid from a SWIFT dataset

    """

    if z_slice is None:
        z_slice = 0.0 * data.gas.coordinates.units

    number_of_gas_particles = data.gas.coordinates.shape[0]

    if project is None:
        m = ones(number_of_gas_particles, dtype=float32)
    else:
        m = getattr(data.gas, project)
        if data.gas.coordinates.comoving:
            if not m.compatible_with_comoving():
                raise AttributeError(
                    f'Physical quantity "{project}" is not compatible with comoving coordinates!'
                )
        else:
            if not m.compatible_with_physical():
                raise AttributeError(
                    f'Comoving quantity "{project}" is not compatible with physical coordinates!'
                )
        m = m.value

    box_x, box_y, box_z = data.metadata.boxsize

    if z_slice > box_z or z_slice < (0 * box_z):
        raise ValueError("Please enter a slice value inside the box.")

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

    # Deal with non-cubic boxes:
    # we always use the maximum of x_range and y_range to normalise the coordinates
    # empty pixels in the resulting square image are trimmed afterwards
    max_range = max(x_range, y_range)

    if rotation_center is not None:
        # Rotate co-ordinates as required
        x, y, z = matmul(rotation_matrix, (data.gas.coordinates - rotation_center).T)

        x += rotation_center[0]
        y += rotation_center[1]
        z += rotation_center[2]

        z_center = rotation_center[2]

    else:
        x, y, z = data.gas.coordinates.T

        z_center = 0 * box_z

    hsml = backends_get_hsml[backend](data)
    if data.gas.coordinates.comoving:
        if not hsml.compatible_with_comoving():
            raise AttributeError(
                f"Physical smoothing length is not compatible with comoving coordinates!"
            )
    else:
        if not hsml.compatible_with_physical():
            raise AttributeError(
                f"Comoving smoothing length is not compatible with physical coordinates!"
            )

    if periodic:
        periodic_box_x = box_x / max_range
        periodic_box_y = box_y / max_range
        periodic_box_z = box_z / max_range
    else:
        periodic_box_x = 0.0
        periodic_box_y = 0.0
        periodic_box_z = 0.0

    # determine the effective number of pixels for each dimension
    xres = int(resolution * x_range / max_range)
    yres = int(resolution * y_range / max_range)

    common_parameters = dict(
        x=(x - x_min) / max_range,
        y=(y - y_min) / max_range,
        z=z / max_range,
        m=m,
        h=hsml / max_range,
        z_slice=(z_center + z_slice) / max_range,
        xres=xres,
        yres=yres,
        box_x=periodic_box_x,
        box_y=periodic_box_y,
        box_z=periodic_box_z,
    )

    if parallel:
        image = backends_parallel[backend](**common_parameters)
    else:
        image = backends[backend](**common_parameters)

    return image


def slice_gas(
    data: SWIFTDataset,
    resolution: int,
    z_slice: Optional[unyt_quantity] = None,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    rotation_matrix: Union[None, array] = None,
    rotation_center: Union[None, unyt_array] = None,
    region: Union[None, unyt_array] = None,
    backend: str = "sph",
    periodic: bool = True,
):
    """
    Creates a 2D slice of a SWIFT dataset, weighted by data field

    Parameters
    ----------
    data : SWIFTDataset
        Dataset from which slice is extracted

    resolution : int
        Specifies size of return array

    z_slice : unyt_quantity
        Specifies the location along the z-axis where the slice is to be
        extracted, relative to the rotation center or the origin of the box
        if no rotation center is provided. If the perspective is rotated
        this value refers to the location along the rotated z-axis.

    project : str, optional
        Data field to be projected. Default is mass. If None then simply
        count number of particles

    parallel : bool, optional
        used to determine if we will create the image in parallel. This
        defaults to False, but can speed up the creation of large images
        significantly at the cost of increased memory usage.

    rotation_matrix: np.array, optional
        Rotation matrix (3x3) that describes the rotation of the box around
        ``rotation_center``. In the default case, this provides a slice
        perpendicular to the z axis.

    rotation_center: np.array, optional
        Center of the rotation. If you are trying to rotate around a galaxy, this
        should be the most bound particle.

    region : array, optional
        determines where the image will be created
        (this corresponds to the left and right-hand edges, and top and bottom edges)
        if it is not None. It should have a length of four, and take the form:

        [x_min, x_max, y_min, y_max]

        Particles outside of this range are still considered if their
        smoothing lengths overlap with the range.

    backend : str, optional
        Backend to use. Choices are "sph" for interpolation using kernel weights or
        "nearest_neighbours" for nearest neighbour interpolation.

    periodic : bool, optional
        Account for periodic boundaries for the simulation box?
        Default is ``True``.

    Returns
    -------
    ndarray of float32
        a `resolution` x `resolution` array of the contribution
        of the projected data field to the voxel grid from all of the particles

    See Also
    --------
    slice_gas_pixel grid : Creates a 2D slice of a SWIFT dataset
    render_gas : Creates a 3D voxel grid of a SWIFT dataset with appropriate units

    Notes
    -----
    This is a wrapper function for slice_gas_pixel_grid ensuring that output units are
    appropriate
    """

    if z_slice is None:
        z_slice = 0.0 * data.gas.coordinates.units

    image = slice_gas_pixel_grid(
        data,
        resolution,
        z_slice,
        project,
        parallel,
        rotation_matrix,
        rotation_center,
        region,
        backend,
        periodic,
    )

    if region is not None:
        x_range = region[1] - region[0]
        y_range = region[3] - region[2]
        max_range = max(x_range, y_range)
        units = 1.0 / (max_range ** 3)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(
            1.0 / (x_range.units * y_range.units * data.metadata.boxsize.units)
        )
    else:
        max_range = max(data.metadata.boxsize[0], data.metadata.boxsize[1])
        units = 1.0 / (max_range ** 3)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / data.metadata.boxsize.units ** 3)

    comoving = data.gas.coordinates.comoving
    coord_cosmo_factor = data.gas.coordinates.cosmo_factor
    if project is not None:
        units *= getattr(data.gas, project).units
        project_cosmo_factor = getattr(data.gas, project).cosmo_factor
        new_cosmo_factor = project_cosmo_factor / coord_cosmo_factor ** 3
    else:
        new_cosmo_factor = coord_cosmo_factor ** (-3)

    return cosmo_array(
        image, units=units, cosmo_factor=new_cosmo_factor, comoving=comoving
    )
