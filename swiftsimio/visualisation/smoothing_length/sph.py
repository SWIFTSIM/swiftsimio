from swiftsimio import SWIFTDataset, cosmo_array


def get_hsml(data: SWIFTDataset) -> cosmo_array:
    """
    Extract the smoothing lengths from the gas particles (used for slicing).

    Parameters
    ----------
    data : SWIFTDataset
        The Dataset from which slice will be extracted

    Returns
    -------
    The extracted smoothing lengths.
    """
    try:
        hsml = data.gas.smoothing_lengths
    except AttributeError:
        # Backwards compatibility
        hsml = data.gas.smoothing_length
    return hsml
