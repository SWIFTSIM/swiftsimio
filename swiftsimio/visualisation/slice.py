"""Sub-module for slice plots in SWFITSIMio."""

from typing import Union, Optional
import numpy as np
from swiftsimio import SWIFTDataset, cosmo_array, cosmo_quantity
from swiftsimio.visualisation.slice_backends import backends, backends_parallel
from swiftsimio.visualisation.smoothing_length import backends_get_hsml
from swiftsimio.visualisation._vistools import (
    _get_projection_field,
    _get_region_info,
    _get_rotated_and_wrapped_coordinates,
    backend_restore_cosmo_and_units,
)


def slice_gas(
    data: SWIFTDataset,
    resolution: int,
    z_slice: Optional[cosmo_quantity] = None,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    rotation_matrix: Union[None, np.array] = None,
    rotation_center: Union[None, cosmo_array] = None,
    region: Union[None, cosmo_array] = None,
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
        Specifies size of return np.array

    z_slice : cosmo_quantity
        Specifies the location along the z-axis where the slice is to be
        extracted, relative to the rotation center or the origin of the box
        if no rotation center is provided. If the perspective is rotated
        this value refers to the location along the rotated z-axis.

    project : str, optional
        Data field to be projected. Default is mass. If ``None`` then simply
        count number of particles. The result is comoving if this is comoving,
        else it is physical.

    parallel : bool
        used to determine if we will create the image in parallel. This
        defaults to False, but can speed up the creation of large images
        significantly at the cost of increased memory usage.

    rotation_matrix: np.np.array, optional
        Rotation matrix (3x3) that describes the rotation of the box around
        ``rotation_center``. In the default case, this provides a slice
        perpendicular to the z axis.

    rotation_center: np.np.array, optional
        Center of the rotation. If you are trying to rotate around a galaxy, this
        should be the most bound particle.

    region : cosmo_array, optional
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
    image : cosmo_array
        Slice image with units of project / length^2, of size ``res`` x ``res``.
        Comoving if ``project`` data are comoving, else physical.

    See Also
    --------
    render_gas_voxel_grid : Creates a 3D voxel grid from a SWIFT dataset

    """
    data = data.gas
    z_slice = np.zeros_like(data.metadata.boxsize[0]) if z_slice is None else z_slice

    m = _get_projection_field(data, project)
    region_info = _get_region_info(data, region, periodic=periodic)
    hsml = backends_get_hsml[backend](data)
    x, y, z = _get_rotated_and_wrapped_coordinates(
        data, rotation_matrix, rotation_center, periodic
    )
    z_center = (
        rotation_center[2]
        if rotation_center is not None
        else np.zeros_like(data.metadata.boxsize[2])
    )

    # determine the effective number of pixels for each dimension
    xres = int(np.ceil(resolution * region_info["x_range"] / region_info["max_range"]))
    yres = int(np.ceil(resolution * region_info["y_range"] / region_info["max_range"]))

    normed_x = (x - region_info["x_min"]) / region_info["max_range"]
    normed_y = (y - region_info["y_min"]) / region_info["max_range"]
    normed_z = z / region_info["max_range"]
    normed_z_slice = (z_slice + z_center) / region_info["max_range"]
    if periodic:
        # place everything in the region inside [0, 1], the backend will tile as needed
        normed_x %= region_info["periodic_box_x"]
        normed_y %= region_info["periodic_box_y"]
        normed_z %= region_info["periodic_box_z"]
        normed_z_slice %= region_info["periodic_box_z"]
    kwargs = dict(
        x=normed_x,
        y=normed_y,
        z=normed_z,
        m=m,
        h=hsml / region_info["max_range"],
        z_slice=normed_z_slice,
        xres=xres,
        yres=yres,
        box_x=region_info["periodic_box_x"],
        box_y=region_info["periodic_box_y"],
        box_z=region_info["periodic_box_z"],
    )
    norm = region_info["max_range"] ** 3
    backend_func = (backends_parallel if parallel else backends)[backend]
    image = backend_restore_cosmo_and_units(backend_func, norm=norm)(**kwargs)
    return image
