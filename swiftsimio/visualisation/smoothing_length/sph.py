"""Tools to get SPH smoothing lengths."""

from swiftsimio import SWIFTDataset, cosmo_array


def get_hsml(data: SWIFTDataset) -> cosmo_array:
    """
    Extract the smoothing lengths from the gas particles (used for slicing).

    Parameters
    ----------
    data : SWIFTDataset
        The Dataset from which slice will be extracted.

    Returns
    -------
    out : cosmo_array
        The extracted smoothing lengths.
    """
    return (
        data.smoothing_lengths
        if hasattr(data, "smoothing_lengths")
        else data.smoothing_length  # backwards compatibility
    )
