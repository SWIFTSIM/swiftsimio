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
    _get_region_limits,
    _get_rotated_coordinates,
)

scatter = backends["fast"]
scatter_parallel = backends_parallel["fast"]


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
        * data.gas.masses``.

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


    Notes
    -----

    + Particles outside of this range are still considered if their smoothing
      lengths overlap with the range.
    + The returned array has x as the first component and y as the second component,
      which is the opposite to what ``imshow`` requires. You should transpose the
      array if you want it to be visualised the 'right way up'.
    """

    m = _get_projection_field(data, project)

    # This provides a default 'slice it all' mask.
    if mask is None:
        mask = np.s_[:]

    x_min, x_max, y_min, y_max, z_min, z_max, x_range, y_range, _, max_range = _get_region_limits(
        data, region
    )

    if backend == "histogram":
        hsml = np.empty_like(m)  # not used anyway for this backend
    else:
        hsml = backends_get_hsml["sph"](data)

    x, y, z = _get_rotated_coordinates(data, rotation_matrix, rotation_center)

    # ------------------------------

    if (region is not None) and len(
        region
    ) != 6:  # if not z_slice_included: ...should refactor
        combined_mask = np.logical_and(
            mask, np.logical_and(z <= z_max, z >= z_min)
        ).astype(bool)
    else:
        combined_mask = mask

    if periodic:
        periodic_box_x, periodic_box_y, _ = data.metadata.boxsize / max_range
    else:
        periodic_box_x, periodic_box_y = 0.0, 0.0

    common_arguments = dict(
        x=(x[combined_mask] - x_min) / max_range,
        y=(y[combined_mask] - y_min) / max_range,
        m=m[combined_mask],
        h=hsml[combined_mask] / max_range,
        res=resolution,
        box_x=periodic_box_x,
        box_y=periodic_box_y,
    )

    if parallel:
        image = backends_parallel[backend](**common_arguments)
    else:
        image = backends[backend](**common_arguments)

    # determine the effective number of pixels for each dimension
    xres = int(np.ceil(resolution * (x_range / max_range)))
    yres = int(np.ceil(resolution * (y_range / max_range)))

    # trim the image to remove empty pixels
    return image[:xres, :yres]


def project_gas_pixel_grid(
    data: SWIFTDataset,
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

    This function is the same as ``project_gas`` but does not include units.

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
        * data.gas.masses``.

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

    image: np.array
        Projected image with dimensions of project / length^2, of size
        ``res`` x ``res``.


    Notes
    -----

    + Particles outside of this range are still considered if their smoothing
      lengths overlap with the range.
    + The returned array has x as the first component and y as the second component,
      which is the opposite to what ``imshow`` requires. You should transpose the
      array if you want it to be visualised the 'right way up'.
    """

    image = project_pixel_grid(
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

    return image


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
        * data.gas.masses``.

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
        Projected image with units of project / length^2, of size ``res`` x
        ``res``.


    Notes
    -----

    + Particles outside of this range are still considered if their smoothing
      lengths overlap with the range.
    + The returned array has x as the first component and y as the second component,
      which is the opposite to what ``imshow`` requires. You should transpose the
      array if you want it to be visualised the 'right way up'.
    """

    image = project_gas_pixel_grid(
        data=data,
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

    if region is not None:
        x_range = region[1] - region[0]
        y_range = region[3] - region[2]
        max_range = max(x_range, y_range)
        units = 1.0 / (max_range ** 2)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / (x_range.units * y_range.units))
    else:
        max_range = max(data.metadata.boxsize[0], data.metadata.boxsize[1])
        units = 1.0 / (max_range ** 2)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / data.metadata.boxsize.units ** 2)

    comoving = data.gas.coordinates.comoving
    coord_cosmo_factor = data.gas.coordinates.cosmo_factor
    if project is not None:
        units *= getattr(data.gas, project).units
        project_cosmo_factor = getattr(data.gas, project).cosmo_factor
        new_cosmo_factor = project_cosmo_factor / coord_cosmo_factor ** 2
    else:
        new_cosmo_factor = coord_cosmo_factor ** (-2)

    return cosmo_array(
        image, units=units, cosmo_factor=new_cosmo_factor, comoving=comoving
    )
