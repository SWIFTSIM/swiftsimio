"""
Basic volume render for SPH data. This takes the 3D positions
of the particles and projects them onto a grid.
"""

from typing import List, Literal, Tuple, Union
from math import sqrt, exp, pi
import numpy as np
from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.accelerated import jit

from swiftsimio.optional_packages import plt

from swiftsimio.visualisation.smoothing_length import backends_get_hsml
from swiftsimio.visualisation.volume_render_backends import backends, backends_parallel
from swiftsimio.visualisation._vistools import (
    _get_projection_field,
    _get_region_info,
    _get_rotated_and_wrapped_coordinates,
    backend_restore_cosmo_and_units,
)


def render_gas(
    data: SWIFTDataset,
    resolution: int,
    project: Union[str, None] = "masses",
    parallel: bool = False,
    rotation_matrix: Union[None, np.array] = None,
    rotation_center: Union[None, cosmo_array] = None,
    region: Union[None, cosmo_array] = None,
    periodic: bool = True,
):
    """
    Creates a 3D render of a SWIFT dataset, weighted by data field, in the
    form of a voxel grid.

    Parameters
    ----------
    data : SWIFTDataset
        Dataset from which render is extracted

    resolution : int
        Specifies size of return np.array

    project : str, optional
        Data field to be projected. Default is ``"mass"``. If ``None`` then simply
        count number of particles. The result is comoving if this is comoving, else
        it is physical.

    parallel : bool
        used to determine if we will create the image in parallel. This
        defaults to False, but can speed up the creation of large images
        significantly at the cost of increased memory usage.

    rotation_matrix: np.np.array, optional
        Rotation matrix (3x3) that describes the rotation of the box around
        ``rotation_center``. In the default case, this provides a volume render
        viewed along the z axis.

    rotation_center: cosmo_array, optional
        Center of the rotation. If you are trying to rotate around a galaxy, this
        should be the most bound particle.

    region : cosmo_array, optional
        determines where the image will be created
        (this corresponds to the left and right-hand edges, and top and bottom
        edges, and front and back edges) if it is not None. It should have a
        length of six, and take the form:

        [x_min, x_max, y_min, y_max, z_min, z_max]

        Particles outside of this range are still considered if their
        smoothing lengths overlap with the range.

    periodic : bool, optional
        Account for periodic boundaries for the simulation box?
        Default is ``True``.

    Returns
    -------
    cosmo_array
        Voxel grid with units of project / length^3, of size ``resolution`` x
        ``resolution`` x ``resolution``. Comoving if ``project`` data are
        comoving, else physical.

    See Also
    --------
    slice_gas_pixel_grid : Creates a 2D slice of a SWIFT dataset

    """
    data = data.gas

    m = _get_projection_field(data, project)
    region_info = _get_region_info(data, region, require_cubic=True, periodic=periodic)
    hsml = backends_get_hsml["sph"](data)
    x, y, z = _get_rotated_and_wrapped_coordinates(
        data, rotation_matrix, rotation_center, periodic
    )

    normed_x = (x - region_info["x_min"]) / region_info["x_range"]
    normed_y = (y - region_info["y_min"]) / region_info["y_range"]
    normed_z = (z - region_info["z_min"]) / region_info["z_range"]
    if periodic:
        # place everything in the region inside [0, 1], the backend will tile as needed
        normed_x %= region_info["periodic_box_x"]
        normed_y %= region_info["periodic_box_y"]
        normed_z %= region_info["periodic_box_z"]
    kwargs = dict(
        x=normed_x,
        y=normed_y,
        z=normed_z,
        m=m,
        h=hsml / region_info["x_range"],  # cubic so x_range == y_range == z_range
        res=resolution,
        box_x=region_info["periodic_box_x"],
        box_y=region_info["periodic_box_y"],
        box_z=region_info["periodic_box_z"],
    )
    norm = region_info["x_range"] * region_info["y_range"] * region_info["z_range"]
    backend_func = (backends_parallel if parallel else backends)["scatter"]
    image = backend_restore_cosmo_and_units(backend_func, norm=norm)(**kwargs)

    return image


@jit(nopython=True, fastmath=True)
def render_voxel_to_array(data, center, width):
    output = np.zeros((data.shape[0], data.shape[1]))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out = 0.0
            for k in range(data.shape[2]):
                inner = (center - data[i, j, k]) / width
                const = 1.0 / (width * sqrt(2.0 * pi))
                fac = exp(-0.5 * inner * inner)

                out += fac * const
            output[j, i] = out

    return output


def visualise_render(
    render: np.ndarray,
    centers: List[float],
    widths: Union[List[float], float],
    cmap: str = "viridis",
    return_type: Literal["all", "lighten", "add"] = "lighten",
    norm: Union[List["plt.Normalize"], "plt.Normalize", None] = None,
) -> Tuple[Union[List[np.ndarray], np.ndarray], List["plt.Normalize"]]:
    """
    Visualises a render with multiple centers and widths.

    Parameters
    ----------

    render : np.np.array
        The render to visualise. You should scale this appropriately
        before using this function (e.g. use a logarithmic transform!)
        and pass in the 'value' np.array, not the original cosmo_array or
        unyt_array.

    centers : list[float]
        The centers of your rendering functions

    widths: list[float] | float
        The widths of your rendering functions. If a single float, all functions
        will have the same width.

    cmap : str
        The colormap to use for the rendering functions.

    return_type : Literal["all", "lighten", "add"]
        The type of return. If "all", all images are returned. If "lighten",
        the maximum of all images is returned. If "add", the sum of all images
        is returned.

    norm : list[plt.Normalize] | plt.Normalize | None
        The normalisation to use for the rendering functions. If a single
        normalisation, all functions will use the same normalisation.

    Returns
    -------

    list[np.np.array] | np.np.array
        The images of the rendering functions. If return_type is "all", this
        will be a list of images. If return_type is "lighten" or "add", this
        will be a single image.

    list[plt.Normalize]
        The normalisations used for the rendering functions.
    """

    if isinstance(widths, float):
        widths = [widths] * len(centers)

    if norm is None:
        norm = [plt.Normalize() for _ in centers]
    elif not isinstance(norm, list):
        norm = [norm] * len(centers)

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(centers)))[:, :3]

    images = [
        n(render_voxel_to_array(render, center, width))
        for n, center, width in zip(norm, centers, widths)
    ]

    images = [
        np.array([color[0] * x, color[1] * x, color[2] * x]).T
        for color, x in zip(colors, images)
    ]

    if return_type == "all":
        return images, norm

    if return_type == "lighten":
        return np.max(images, axis=0), norm

    if return_type == "add":
        return sum(images), norm


def visualise_render_options(
    centers: List[float], widths: Union[List[float], float], cmap: str = "viridis"
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Creates a figure of your rendering options. The y-axis is the output value
    of the rendering function. The x-axis is your input quantity. You may wish
    to plot a histogram on top of this figure; this is why the figure axes and
    figure are returned.

    Parameters
    ----------

    centers : list[float]
        The centers of your rendering functions

    widths : list[float] | float
        The widths of your rendering functions. If a single float, all functions
        will have the same width.

    cmap : str
        The colormap to use for the rendering functions.

    Returns
    -------

    plt.Figure, plt.Axes
        The figure and axes of the plot.
    """
    fig, ax = plt.subplots()

    if isinstance(widths, float):
        widths = [widths] * len(centers)

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(centers)))[:, :3]

    for center, width, color in zip(centers, widths, colors):
        xs = np.linspace(center - 5.0 * width, center + 5.0 * width, 100)
        ys = [
            exp(-0.5 * ((center - x) / width) ** 2) / (width * sqrt(2.0 * pi))
            for x in xs
        ]

        ax.axvline(center, color=color, linestyle="--")

        ax.plot(xs, ys, color=color)

    return fig, ax
