"""Define name mappings for SOAP files."""


# Describes the conversion of hdf5 groups to names
def get_soap_name_underscore(group: str) -> str:
    """
    Define conversions from HDF5 groups to names.

    Parameters
    ----------
    group : str
        The name of the HDF5 group.

    Returns
    -------
    str
        The name to use in swiftsimio.
    """
    soap_name_underscores = {
        "BoundSubhalo": "bound_subhalo",
        "InputHalos": "input_halos",
        "InclusiveSphere": "inclusive_sphere",
        "ExclusiveSphere": "exclusive_sphere",
        "SO": "spherical_overdensity",
        "SOAP": "soap",
        "ProjectedAperture": "projected_aperture",
    }
    split_name = group.split("/")
    split_name[0] = soap_name_underscores[split_name[0]]
    return "_".join(name.lower() for name in split_name)


def get_soap_name_nice(group: str) -> str:
    """
    Define conversions from HDF5 groups to nice names (less acronyms).

    Parameters
    ----------
    group : str
        The name of the HDF5 group.

    Returns
    -------
    str
        The nice name.
    """
    soap_name_nice = {
        "BoundSubhalo": "BoundSubhalo",
        "InputHalos": "InputHalos",
        "InclusiveSphere": "InclusiveSphere",
        "ExclusiveSphere": "ExclusiveSphere",
        "SO": "SphericalOverdensity",
        "SOAP": "SOAP",
        "ProjectedAperture": "ProjectedAperture",
    }
    split_name = group.split("/")
    split_name[0] = soap_name_nice[split_name[0]]
    return "".join(name.capitalize() for name in split_name)
