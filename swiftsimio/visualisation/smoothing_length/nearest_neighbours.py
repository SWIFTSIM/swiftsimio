"""Tools to get nearest-neighbour smoothing lengths."""

import numpy as np
from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.visualisation.smoothing_length.sph import get_hsml as get_hsml_sph


def get_hsml(data: SWIFTDataset) -> cosmo_array:
    """
    Compute a "smoothing length".

    Estimated as the 3rd root of the volume of the particles. This scheme uses volume
    weighting when computing slices.

    Parameters
    ----------
    data : SWIFTDataset
        The dataset from which slice will be extracted.

    Returns
    -------
    cosmo_array
        The extracted "smoothing lengths".
    """
    try:
        hsml = np.cbrt(data.gas.volumes)
    except AttributeError:
        try:
            # Try computing the volumes explicitly?
            hsml = np.cbrt(data.gas.masses / data.gas.densities)
        except AttributeError:
            # Fall back to SPH behavior if above didn't work...
            hsml = get_hsml_sph(data)
    return hsml
