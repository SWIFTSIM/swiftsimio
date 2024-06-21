"""
Includes the fancy names.
"""

# Describes the conversion of hdf5 groups to names
def get_soap_name_underscore(group: str) -> str:
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
    soap_name_nice = {
        "BoundSubhalo": "BoundSubhalo",
        "InputHalos": "InputHalos",
        "InclusiveSphere": "InclusiveSphere",
        "ExclusiveSphere": "ExclusiveSphere",
        "SO": "SphericalOverdensity",
        "SOAP": "SOAP",
        "ProjectedAperture": "ProjectedAperture",
    }
    split_name = group.split('/')
    return ''.join(name.capitalize() for name in split_name)

