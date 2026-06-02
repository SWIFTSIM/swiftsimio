"""Tools to get nearest-neighbour smoothing lengths."""

import numpy as np
from swiftsimio import cosmo_array
from swiftsimio.reader import __SWIFTGroupDataset
from swiftsimio.visualisation.smoothing_length.sph import get_hsml as get_hsml_sph


def get_hsml(data: __SWIFTGroupDataset) -> cosmo_array:
    """
    Compute a "smoothing length".

    Estimated as the 3rd root of the volume of the particles. This scheme uses volume
    weighting when computing slices.

    Parameters
    ----------
    data : __SWIFTGroupDataset
        The particle dataset for which smoothing lengths will be computed.

    Returns
    -------
    cosmo_array
        The extracted "smoothing lengths".
    """
    if hasattr(data, "volumes"):
        hsml = np.cbrt(data.volumes)
    elif hasattr(data, "masses") and hasattr(data, "densities"):
        # Try computing the volumes explicitly?
        hsml = np.cbrt(data.masses / data.densities)
    else:
        # Fall back to SPH behavior if above didn't work...
        hsml = get_hsml_sph(data)
    return hsml
