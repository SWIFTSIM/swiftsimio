"""
Calls functions from `projection_backends`.
"""

from typing import Union
from math import sqrt
from numpy import (
    float64,
    float32,
    int32,
    zeros,
    array,
    arange,
    ndarray,
    ones,
    isclose,
    matmul,
    s_,
)
from unyt import unyt_array, unyt_quantity
from swiftsimio import SWIFTDataset

from swiftsimio.reader import __SWIFTParticleDataset
from swiftsimio.accelerated import jit, NUM_THREADS, prange

from swiftsimio.visualisation.projection_backends import backends, backends_parallel

# Backwards compatability

from swiftsimio.visualisation.projection_backends.kernels import (
    kernel_gamma,
    kernel_constant,
)
from swiftsimio.visualisation.projection_backends.kernels import (
    kernel_single_precision as kernel,
)

scatter = backends["fast"]
scatter_parallel = backends_parallel["fast"]


def project_pixel_grid(
    data: __SWIFTParticleDataset,
    boxsize: unyt_array,
    resolution: int,
    project: Union[str, None] = "masses",
    region: Union[None, unyt_array] = None,
    mask: Union[None, array] = None,
    rotation_matrix: Union[None, array] = None,
    rotation_center: Union[None, unyt_array] = None,
    parallel: bool = False,
    backend: str = "fast",
):
    r"""
    Creates a 2D projection of a SWIFT dataset, projected by the "project"
    variable (e.g. if project is Temperature, we return: \bar{T} = \sum_j T_j
    W_{ij}).

    Default projection variable is mass. If it is None, then we don't
    weight with anything, providing a number density image.

    Parameters
    ----------

    data: __SWIFTParticleDataset
        The SWIFT dataset that you wish to visualise (get this from ``load``)

    boxsize: unyt_array
        The box-size of the simulation.

    resolution: int
        The resolution of the image. All images returned are square, ``res``
        by ``res``, pixel grids.

    project: str, optional
        Variable to project to get the weighted density of. By default, this
        is mass. If you would like to mass-weight any other variable, you can
        always create it as ``data.gas.my_variable = data.gas.other_variable
        * data.gas.masses``.

    region: unyt_array, optional
        Region, determines where the image will be created (this corresponds
        to the left and right-hand edges, and top and bottom edges) if it is
        not None. It should have a length of four, and take the form:
        ``[x_min, x_max, y_min, y_max]``

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


    Returns
    -------

    image: unyt_array
        Projected image with units of project / length^2, of size ``res`` x ``res``.


    Notes
    -----

    + Particles outside of this range are still considered if their smoothing
      lengths overlap with the range.
    + The returned array has x as the first component and y as the second component,
      which is the opposite to what ``imshow`` requires. You should transpose the
      array if you want it to be visualised the 'right way up'.
    """

    number_of_particles = data.coordinates.shape[0]

    if project is None:
        m = ones(number_of_particles, dtype=float32)
    else:
        m = getattr(data, project).value

    # This provides a default 'slice it all' mask.
    if mask is None:
        mask = s_[:]

    box_x, box_y, _ = boxsize

    # Set the limits of the image.
    if region is not None:
        x_min, x_max, y_min, y_max = region
    else:
        x_min = unyt_quantity(0.0, units=box_x.units)
        x_max = box_x
        y_min = unyt_quantity(0.0, units=box_y.units)
        y_max = box_y

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Test that we've got a square box
    if not isclose(x_range.value, y_range.value):
        raise AttributeError(
            "Projection code is currently not able to handle non-square images"
        )

    try:
        hsml = data.smoothing_lengths
    except AttributeError:
        # Backwards compatibility
        hsml = data.smoothing_length

    if rotation_center is not None:
        # Rotate co-ordinates as required
        x, y, _ = matmul(rotation_matrix, (data.coordinates - rotation_center).T)

        x += rotation_center[0]
        y += rotation_center[1]

    else:
        x, y, _ = data.coordinates.T

    common_arguments = dict(
        x=(x[mask] - x_min) / x_range,
        y=(y[mask] - y_min) / y_range,
        m=m[mask],
        h=hsml[mask] / x_range,
        res=resolution,
    )

    if parallel:
        image = backends_parallel[backend](**common_arguments)
    else:
        image = backends[backend](**common_arguments)

    return image


def project_gas_pixel_grid(
    data: SWIFTDataset,
    resolution: int,
    project: Union[str, None] = "masses",
    region: Union[None, unyt_array] = None,
    mask: Union[None, array] = None,
    rotation_matrix: Union[None, array] = None,
    rotation_center: Union[None, unyt_array] = None,
    parallel: bool = False,
    backend: str = "fast",
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

    region: unyt_array, optional
        Region, determines where the image will be created (this corresponds
        to the left and right-hand edges, and top and bottom edges) if it is
        not None. It should have a length of four, and take the form:
        ``[x_min, x_max, y_min, y_max]``

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
        boxsize=data.metadata.boxsize,
        resolution=resolution,
        project=project,
        mask=mask,
        parallel=parallel,
        region=region,
        rotation_matrix=rotation_matrix,
        rotation_center=rotation_center,
        backend=backend,
    )

    return image


def project_gas(
    data: SWIFTDataset,
    resolution: int,
    project: Union[str, None] = "masses",
    region: Union[None, unyt_array] = None,
    mask: Union[None, array] = None,
    rotation_center: Union[None, unyt_array] = None,
    rotation_matrix: Union[None, array] = None,
    parallel: bool = False,
    backend: str = "fast",
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

    region: unyt_array, optional
        Region, determines where the image will be created (this corresponds
        to the left and right-hand edges, and top and bottom edges) if it is
        not None. It should have a length of four, and take the form:
        ``[x_min, x_max, y_min, y_max]``

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


    Returns
    -------

    image: unyt_array
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
    )

    if region is not None:
        x_range = region[1] - region[0]
        y_range = region[3] - region[2]
        units = 1.0 / (x_range * y_range)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / (x_range.units * y_range.units))
    else:
        units = 1.0 / (data.metadata.boxsize[0] * data.metadata.boxsize[1])
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / data.metadata.boxsize.units ** 2)

    if project is not None:
        units *= getattr(data.gas, project).units

    return image * units
