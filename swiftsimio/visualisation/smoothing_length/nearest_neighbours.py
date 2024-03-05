from numpy import cbrt

from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.objects import _cbrt_cosmo_factor
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
        # TODO remove this hack once np.cbrt is supported by unyt
        volumes = data.gas.volumes
        units = (hasattr(volumes, "units"), getattr(volumes, "units", None))
        comoving = getattr(volumes, "comoving", None)
        cosmo_factor = (hasattr(volumes, "cosmo_factor"), getattr(volumes, "cosmo_factor", None))
        if units[0]:
            units_hsml = units[1] ** (1. / 3.)
        else:
            units_hsml = None
        hsml = cosmo_array(
            cbrt(volumes.value),
            units=units_hsml,
            comoving=comoving,
            cosmo_factor=_cbrt_cosmo_factor(cosmo_factor),
        )
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
