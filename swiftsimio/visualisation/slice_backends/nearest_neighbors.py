from numpy import float64, float32, ndarray, linspace, array, stack, meshgrid, cbrt

from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.optional_packages import KDTree, TREE_AVAILABLE
from swiftsimio.visualisation.slice_backends.sph import get_hsml as sph_get_hsml


def build_tree(
    x: float64, y: float64, z: float64, box_x: float, box_y: float, box_z: float
) -> KDTree:
    """
    Build the tree used for the nearest-neighbor calculations.
    In the periodic case, we must make sure that all particle coordinates
    fall inside the box.

    Parameters
    ----------
    x : array of float64
        x-positions of the particles. Must be bounded by [0, 1].
    y : array of float64
        y-positions of the particles. Must be bounded by [0, 1].
    z : array of float64
        z-positions of the particles. Must be bounded by [0, 1].
    box_x: float
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_y: float
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_z: float
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).

    Returns
    -------
    KDTree object
        A KD-tree built from the particle positions
    """
    if not TREE_AVAILABLE:
        raise ImportError(
            "The scipy.spatial.cKDTree class is required to use the "
            "'nearest_neighbors' slice backend."
        )
    if box_x != 0 or box_y != 0 or box_z != 0:
        if box_x != 0:
            x[x < 0] += box_x
        if box_y != 0:
            y[y < 0] += box_y
        if box_z != 0:
            z[z < 0] += box_z
        data = stack((x, y, z), axis=1)
        return KDTree(data, boxsize=[box_x, box_y, box_z])
    else:
        data = stack((x, y, z), axis=1)
        return KDTree(data)


def get_hsml(data: SWIFTDataset) -> cosmo_array:
    """
    Computes a "smoothing length" as the 3rd root of the volume of the particles.
    This scheme uses volume weighting when computing slices.

    Parameters
    ----------
    data : SWIFTDataset
        The Dataset from which slice will be extracted

    Returns
    -------
    The extracted "smoothing lengths".
    """
    try:
        hsml = cbrt(data.gas.volume)
    except AttributeError:
        try:
            # Try computing the volumes explicitly?
            masses = data.gas.masses
            densities = data.gas.densities
            hsml = cbrt(masses / densities)
        except AttributeError:
            # Fall back to SPH behavior if above didn't work...
            hsml = sph_get_hsml(data)
    return hsml


def slice_scatter(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float,
    xres: int,
    yres: int,
    box_x: float = 0.0,
    box_y: float = 0.0,
    box_z: float = 0.0,
    workers: int = 1,
) -> ndarray:
    """
    Creates a scatter plot of the given quantities for a particles in a data slice.

    Parameters
    ----------
    x : array of float64
        x-positions of the particles. Must be bounded by [0, 1].
    y : array of float64
        y-positions of the particles. Must be bounded by [0, 1].
    z : array of float64
        z-positions of the particles. Must be bounded by [0, 1].
    m : array of float32
        masses (or otherwise weights) of the particles
    h : array of float32
        smoothing lengths of the particles
    z_slice : float64
        the position at which we wish to create the slice
    xres : int
        the number of pixels in x direction.
    yres : int
        the number of pixels in the y direction.
    box_x: float
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_y: float
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_z: float
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    workers : int
        The number of workers to use for the nearest-neighbor calculations.
        Set to -1 to use all available cpus.

    Returns
    -------
    ndarray of float32
        output array for scatterplot image

    See Also
    --------
    scatter : Create 3D scatter plot of SWIFT data
    scatter_parallel : Create 3D scatter plot of SWIFT data in parallel
    slice_scatter_parallel : Create scatter plot of a slice of data in parallel
    """
    res = max(xres, yres)
    pixel_coordinates = stack(
        [
            arr.ravel()
            for arr in meshgrid(
                linspace(0, xres / res, xres) + 0.5 / res,
                linspace(0, yres / res, yres) + 0.5 / res,
                array([z_slice]),
                indexing="ij",
            )
        ],
        axis=1,
    )

    tree = build_tree(x, y, z, box_x, box_y, box_z)
    _, i = tree.query(pixel_coordinates, workers=workers)

    values = m / h ** 3
    return values[i].reshape(xres, yres)


def slice_scatter_parallel(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float,
    xres: int,
    yres: int,
    box_x: float = 0.0,
    box_y: float = 0.0,
    box_z: float = 0.0,
) -> ndarray:
    """
    Parallel implementation of slice_scatter

    Creates a scatter plot of the given quantities for a particles in a data slice including periodic boundary effects.

    Parameters
    ----------
    x : array of float64
        x-positions of the particles. Must be bounded by [0, 1].
    y : array of float64
        y-positions of the particles. Must be bounded by [0, 1].
    z : array of float64
        z-positions of the particles. Must be bounded by [0, 1].
    m : array of float32
        masses (or otherwise weights) of the particles
    h : array of float32
        smoothing lengths of the particles
    z_slice : float64
        the position at which we wish to create the slice
    xres : int
        the number of pixels in x direction.
    yres : int
        the number of pixels in the y direction.
    box_x: float
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_y: float
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_z: float
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).

    Returns
    -------
    ndarray of float32
        output array for scatterplot image

    See Also
    --------
    scatter : Create 3D scatter plot of SWIFT data
    scatter_parallel : Create 3D scatter plot of SWIFT data in parallel
    slice_scatter_parallel : Create scatter plot of a slice of data in parallel
    """
    return slice_scatter(
        x=x,
        y=y,
        z=z,
        m=m,
        h=h,
        z_slice=z_slice,
        xres=xres,
        yres=yres,
        box_x=box_x,
        box_y=box_y,
        box_z=box_z,
        workers=-1,
    )
