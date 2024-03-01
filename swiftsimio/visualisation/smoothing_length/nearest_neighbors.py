from numpy import cbrt

from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.visualisation.smoothing_length.sph import get_hsml as get_hsml_sph


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
            hsml = get_hsml_sph(data)
    return hsml
