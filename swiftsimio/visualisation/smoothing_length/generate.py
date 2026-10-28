"""
Smoothing length generators.

Routines for generating (approximate) smoothing lengths for particles
that do not usually carry a smoothing length field (e.g. dark matter).
"""

import numpy as np
from swiftsimio import cosmo_array
from swiftsimio._array_functions import _propagate_cosmo_array_attributes_to_result
from swiftsimio.visualisation.kdtree import k_nearest_distances


@_propagate_cosmo_array_attributes_to_result  # copies attrs of first arg to result
def generate_smoothing_lengths(
    coordinates: cosmo_array,
    boxsize: cosmo_array,
    kernel_gamma: np.float32,
    neighbours: int = 32,
    speedup_fac: int = 2,
    dimension: int = 3,
) -> cosmo_array:
    """
    Generate smoothing lengths that encompass a number of neighbours specified here.

    Parameters
    ----------
    coordinates : cosmo_array
        A cosmo_array that gives the co-ordinates of all particles.

    boxsize : cosmo_array
        The size of the box (3D).

    kernel_gamma : np.float32
        The kernel gamma of the kernel being used.

    neighbours : int, optional
        The number of neighbours to encompass.

    speedup_fac : int, optional
        A parameter that neighbours is divided by to provide a speed-up
        by only searching for a lower number of neighbours. For example,
        if neighbours is 32, and speedup_fac is 2, we only search for 16
        (32 / 2) neighbours, and extend the smoothing length out to
        (speedup)**(1/dimension) such that we encompass an approximately
        higher number of neighbours. A factor of 2 gives smoothing lengths
        the same as the full search within 10%, good enough for visualisation.

    dimension : int, optional
        The dimensionality of the problem (used for speedup_fac calculation).

    Returns
    -------
    cosmo_array
        A cosmo_array of smoothing lengths.

    Notes
    -----
    The neighbours are found with a parallel k-d tree compiled with ``numba``
    (running on ``numba.get_num_threads()`` threads), which gives the same
    distances as :class:`scipy.spatial.cKDTree`. The box is periodic along every
    axis with a positive ``boxsize``.
    """
    # Include speedup_fac stuff here:
    neighbours_search = neighbours // speedup_fac
    hsml_correction_fac_speedup = (speedup_fac) ** (1 / dimension)

    # Distance to the neighbours_search-th nearest neighbour (the particle itself
    # included), with a parallel numba k-d tree: the same distances as
    # scipy.spatial.cKDTree.query
    smoothing_lengths = k_nearest_distances(
        coordinates.to_value(coordinates.units),
        neighbours_search,
        boxsize=boxsize.to_value(coordinates.units),
    ).astype(np.float32)

    return type(coordinates)(
        smoothing_lengths
        * (hsml_correction_fac_speedup / kernel_gamma)
        * coordinates.units
    )
