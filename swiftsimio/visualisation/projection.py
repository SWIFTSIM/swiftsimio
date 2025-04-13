"""
Calls functions from `projection_backends`.
"""

from typing import Union
import numpy as np
from swiftsimio import SWIFTDataset, cosmo_array

from swiftsimio.reader import __SWIFTGroupDataset
from swiftsimio.visualisation.projection_backends import backends, backends_parallel
from swiftsimio.visualisation.smoothing_length import backends_get_hsml
from swiftsimio.visualisation._vistools import (
    _get_projection_field,
    _get_region_info,
    _get_rotated_and_wrapped_coordinates,
    backend_restore_cosmo_and_units,
)


def project_pixel_grid(
    data: __SWIFTGroupDataset,
    resolution: int,
    project: Union[str, None] = "masses",
    region: Union[None, cosmo_array] = None,
    mask: Union[None, np.array] = None,
    rotation_matrix: Union[None, np.array] = None,
    rotation_center: Union[None, cosmo_array] = None,
    parallel: bool = False,
    backend: str = "fast",
    periodic: bool = True,
):
    r"""
    Creates a 2D projection of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return: \bar{T} = \sum_j T_j
    W_{ij}).

    Default projection variable is mass. If it is None, then we don't
    weight with anything, providing a number density image.

    Parameters
    ----------

    data: __SWIFTGroupDataset
        The SWIFT dataset that you wish to visualise (get this from ``load``)

    resolution: int
        The resolution of the image. All images returned are square, ``res``
        by ``res``, pixel grids.

    project: str, optional
        Variable to project to get the weighted density of. By default, this
        is mass. If you would like to mass-weight any other variable, you can
        always create it as ``data.gas.my_variable = data.gas.other_variable
        * data.gas.masses``. The result is comoving if this is comoving, else
        it is physical.

    region: cosmo_array, optional
        Region, determines where the image will be created (this corresponds
        to the left and right-hand edges, and top and bottom edges) if it is
        not None. It should have a length of four or six, and take the form:
        ``[x_min, x_max, y_min, y_max, {z_min, z_max}]``

    mask: np.array, optional
        Allows only a sub-set of the particles in data to be visualised. Useful
        in cases where you have read data out of a ``velociraptor`` catalogue,
        or if you only want to visualise e.g. star forming particles. This boolean
        mask is applied just before visualisation.

    rotation_center: np.array, optional
        Center of the rotation. If you are trying to rotate around a galaxy, this
        should be the most bound particle.

    rotation_matrix: np.array, optional
        Rotation matrix (3x3) that describes the rotation of the box around
        ``rotation_center``. In the default case, this provides a projection
        along the z axis.

    parallel: bool, optional
        Defaults to ``False``, whether or not to create the image in parallel.
        The parallel version of this function uses significantly more memory.

    backend: str, optional
        Backend to use. See documentation for details. Defaults to 'fast'.

    periodic: bool, optional
        Account for periodic boundary conditions for the simulation box?
        Defaults to ``True``.

    Returns
    -------

    image: cosmo_array
        Projected image with units of project / length^2, of size ``res`` x ``res``.
        Comoving if ``project`` data are comoving, else physical.


    Notes
    -----

    + Particles outside of this range are still considered if their smoothing
      lengths overlap with the range.
    + The returned array has x as the first component and y as the second component,
      which is the opposite to what ``imshow`` requires. You should transpose the
      array if you want it to be visualised the 'right way up'.
    """

    m = _get_projection_field(data, project)
    region_info = _get_region_info(data, region, periodic=periodic)
    hsml = backends_get_hsml["sph" if backend != "histogram" else "histogram"](data)
    x, y, z = _get_rotated_and_wrapped_coordinates(
        data, rotation_matrix, rotation_center, periodic
    )
    mask = mask if mask is not None else np.s_[...]
    if not region_info["z_slice_included"]:
        mask = np.logical_and(
            mask, np.logical_and(z <= region_info["z_max"], z >= region_info["z_min"])
        ).astype(bool)

    normed_x = ((x[mask] - region_info["x_min"]) / region_info["max_range"])
    normed_y = ((y[mask] - region_info["y_min"]) / region_info["max_range"])
    if periodic:
        # place everything inside the [0, 1] box, the backend will tile as needed
        normed_x %= 1
        normed_y %= 1
    kwargs = dict(
        x=normed_x,
        y=normed_y,
        m=m[mask],
        h=hsml[mask] / region_info["max_range"],
        res=resolution,
        box_x=region_info["periodic_box_x"],
        box_y=region_info["periodic_box_y"],
    )
    norm = region_info["x_range"] * region_info["y_range"]
    backend_func = (backends_parallel if parallel else backends)[backend]
    image = backend_restore_cosmo_and_units(backend_func, norm=norm)(**kwargs)

    # determine the effective number of pixels for each dimension
    xres = int(
        np.ceil(resolution * (region_info["x_range"] / region_info["max_range"]))
    )
    yres = int(
        np.ceil(resolution * (region_info["y_range"] / region_info["max_range"]))
    )

    # trim the image to remove empty pixels
    return image[:xres, :yres]


def project_gas(
    data: SWIFTDataset,
    resolution: int,
    project: Union[str, None] = "masses",
    region: Union[None, cosmo_array] = None,
    mask: Union[None, np.array] = None,
    rotation_center: Union[None, cosmo_array] = None,
    rotation_matrix: Union[None, np.array] = None,
    parallel: bool = False,
    backend: str = "fast",
    periodic: bool = True,
):
    r"""
    Creates a 2D projection of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return: \bar{T} = \sum_j T_j
    W_{ij}).

    Default projection variable is mass. If it is None, then we don't
    weight with anything, providing a number density image.

    Parameters
    ----------

    data: SWIFTDataset
        The SWIFT dataset that you wish to visualise (get this from ``load``)

    resolution: int
        The resolution of the image. All images returned are square, ``res``
        by ``res``, pixel grids.

    project: str, optional
        Variable to project to get the weighted density of. By default, this
        is mass. If you would like to mass-weight any other variable, you can
        always create it as ``data.gas.my_variable = data.gas.other_variable
        * data.gas.masses``. The result is comoving if this is comoving, else
        it is physical.

    region: cosmo_array, optional
        Region, determines where the image will be created (this corresponds
        to the left and right-hand edges, and top and bottom edges) if it is
        not None. It should have a length of four or six, and take the form:
        ``[x_min, x_max, y_min, y_max, {z_min, z_max}]``

    mask: np.array, optional
        Allows only a sub-set of the particles in data to be visualised. Useful
        in cases where you have read data out of a ``velociraptor`` catalogue,
        or if you only want to visualise e.g. star forming particles. This boolean
        mask is applied just before visualisation.

    rotation_center: np.array, optional
        Center of the rotation. If you are trying to rotate around a galaxy, this
        should be the most bound particle.

    rotation_matrix: np.array, optional
        Rotation matrix (3x3) that describes the rotation of the box around
        ``rotation_center``. In the default case, this provides a projection
        along the z axis.

    parallel: bool, optional
        Defaults to ``False``, whether or not to create the image in parallel.
        The parallel version of this function uses significantly more memory.

    backend: str, optional
        Backend to use. See documentation for details. Defaults to 'fast'.

    periodic: bool, optional
        Account for periodic boundary conditions for the simulation box?
        Defaults to ``True``.


    Returns
    -------

    image: cosmo_array
        Projected image with units of project / length^2, of size ``res`` x ``res``.
        Comoving if ``project`` data are comoving, else physical.


    Notes
    -----

    + Particles outside of this range are still considered if their smoothing
      lengths overlap with the range.
    + The returned array has x as the first component and y as the second component,
      which is the opposite to what ``imshow`` requires. You should transpose the
      array if you want it to be visualised the 'right way up'.
    """

    return project_pixel_grid(
        data=data.gas,
        resolution=resolution,
        project=project,
        mask=mask,
        parallel=parallel,
        region=region,
        rotation_matrix=rotation_matrix,
        rotation_center=rotation_center,
        backend=backend,
        periodic=periodic,
    )
