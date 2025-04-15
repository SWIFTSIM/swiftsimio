"""
Routines for generating (approximate) smoothing lengths for particles
that do not usually carry a smoothing length field (e.g. dark matter).
"""

import numpy as np
from swiftsimio import cosmo_array
from swiftsimio.optional_packages import KDTree, TREE_AVAILABLE
from swiftsimio._array_functions import _propagate_cosmo_array_attributes_to_result


@_propagate_cosmo_array_attributes_to_result  # copies attrs of first arg to result
def generate_smoothing_lengths(
    coordinates: cosmo_array,
    boxsize: cosmo_array,
    kernel_gamma: np.float32,
    neighbours=32,
    speedup_fac=2,
    dimension=3,
):
    """
    Generates smoothing lengths that encompass a number of neighbours specified here.

    Parameters
    ----------
    coordinates : cosmo_array
        a cosmo_array that gives the co-ordinates of all particles
    boxsize : cosmo_array
        the size of the box (3D)
    kernel_gamma : np.float32
        the kernel gamma of the kernel being used
    neighbours : int, optional
        the number of neighbours to encompass
    speedup_fac : int, optional
        a parameter that neighbours is divided by to provide a speed-up
        by only searching for a lower number of neighbours. For example,
        if neighbours is 32, and speedup_fac is 2, we only search for 16
        (32 / 2) neighbours, and extend the smoothing length out to
        (speedup)**(1/dimension) such that we encompass an approximately
        higher number of neighbours. A factor of 2 gives smoothing lengths
        the same as the full search within 10%, good enough for visualisation.
    dimension : int, optional
        the dimensionality of the problem (used for speedup_fac calculation).

    Returns
    -------
    smoothing lengths : cosmo_array
        a cosmo_array of smoothing lengths.
    """

    if not TREE_AVAILABLE:
        raise ImportError(
            "The scipy.spatial.cKDTree class is required to search for smoothing lengths."
        )

    number_of_parts = coordinates.shape[0]

    tree = KDTree(
        coordinates.to_value(coordinates.units),
        boxsize=boxsize.to_value(coordinates.units),
    )

    smoothing_lengths = np.empty(number_of_parts, dtype=np.float32)
    smoothing_lengths[-1] = -0.1

    # Include speedup_fac stuff here:
    neighbours_search = neighbours // speedup_fac
    hsml_correction_fac_speedup = (speedup_fac) ** (1 / dimension)

    # We create a lot of data doing this, so we want to do it in small chunks
    # such that we keep the memory from filling up - this seems to be a reasonable
    # chunk size based on previous performance testing. This may change in the
    # future, or may be computer dependent (cache sizes?).
    block_size = 65536
    number_of_blocks = 1 + number_of_parts // block_size

    for block in range(number_of_blocks):
        starting_index = block * block_size
        ending_index = (block + 1) * (block_size)

        if ending_index > number_of_parts:
            ending_index = number_of_parts + 1

        if starting_index >= ending_index:
            break

        # Get the distances to _all_ neighbours out of the tree - this is
        # why we need to process in blocks (this is 32x+ the size of coordinates)

        try:
            d, _ = tree.query(
                coordinates[starting_index:ending_index].value,
                k=neighbours_search,
                workers=-1,
            )
        except TypeError:
            # Backwards compatibility with older versions of
            # scipy.
            d, _ = tree.query(
                coordinates[starting_index:ending_index].value,
                k=neighbours_search,
                n_jobs=-1,
            )

        smoothing_lengths[starting_index:ending_index] = d[:, -1]

    return type(coordinates)(
        smoothing_lengths
        * (hsml_correction_fac_speedup / kernel_gamma)
        * coordinates.units
    )
