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
    hsml = (
        data.smoothing_lengths
        if hasattr(data, "smoothing_lengths")
        else data.smoothing_length  # backwards compatibility
    )
    if data.coordinates.comoving:
        if not hsml.compatible_with_comoving():
            raise AttributeError(
                "Physical smoothing length is not compatible with comoving coordinates!"
            )
    else:
        if not hsml.compatible_with_physical():
            raise AttributeError(
                "Comoving smoothing length is not compatible with physical coordinates!"
            )
    return hsml
