"""Tools to get SPH smoothing lengths."""

from swiftsimio import cosmo_array
from swiftsimio.reader import __SWIFTGroupDataset


def get_hsml(data: __SWIFTGroupDataset) -> cosmo_array:
    """
    Extract the smoothing lengths from a particle dataset.

    Parameters
    ----------
    data : __SWIFTGroupDataset
        The particle dataset from which smoothing lengths will be extracted.

    Returns
    -------
    cosmo_array
        The extracted smoothing lengths.
    """
    if hasattr(data, "smoothing_lengths"):
        return data.smoothing_lengths
    elif hasattr(data, "smoothing_length"):
        return data.smoothing_length  # backwards compatibility
    else:
        raise AttributeError(
            f"Particle type {data.group_name} does not have smoothing lengths. "
            "Use generate_smoothing_lengths from swiftsimio.visualisation.smoothing_length "
            "to generate them before visualising."
        )
