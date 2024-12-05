"""
Contains the description of the metadata fields in the SWIFT snapshots.
"""

from ..objects import cosmo_factor, a

metadata_fields_to_read = {
    "Code": "code",
    "Cosmology": "cosmology_raw",
    "Header": "header",
    "GravityScheme": "gravity_scheme",
    "HydroScheme": "hydro_scheme",
    "InternalCodeUnits": "internal_code_units",
    "Parameters": "parameters",
    "Policy": "policy",
    "RuntimePars": "runtime_pars",
    "SubgridScheme": "subgrid_scheme",
    "StarsScheme": "stars_scheme",
    "UnusedParameters": "unused_parameters",
}


# These will be unpacked to the top-level object. Be careful not to overwrite
# things in the same namespace!
header_unpack_arrays = {
    "BoxSize": "boxsize",
    "NumPart_ThisFile": "num_part",
    "NumGroup_ThisFile": "num_group",
    "NumSubhalos_ThisFile": "num_subhalo",
    "CanHaveTypes": "has_type",
    "NumFilesPerSnapshot": "num_files_per_snapshot",
    "OutputType": "output_type",
    "SubhaloTypes": "subhalo_types",
}

# Some of these 'arrays' are really types of mass table, so unpack
# those differently:
header_unpack_mass_tables = {
    "MassTable": "mass_table",
    "InitialMassTable": "initial_mass_table",
}


def generate_units_header_unpack_arrays(m, l, t, I, T) -> dict:
    """
    Generates the unit dictionaries with the:

    mass, length, time, current, and temperature

    units respectively.
    """

    # Do not include those items that do not have units.
    units = {"boxsize": l}

    return units


def generate_cosmo_args_header_unpack_arrays(scale_factor) -> dict:
    """
    Generates arguments for `cosmo_array` so that relevant header
    items can be initialized as `cosmo_array`s.

    Parameters
    ----------
    scale_factor : float
        The scale factor.

    Returns
    -------
    out : dict
        A dictionary containing the `cosmo_array` arguments corresponding to
        header items, omitting any that should not be `cosmo_array`s.
    """

    # Do not include those items that do not have units (and therefore
    # should not be cosmo_array'd).
    cosmo_args = {
        "boxsize": dict(
            cosmo_factor=cosmo_factor(a ** 1, scale_factor=scale_factor),
            comoving=True,  # if it's not, then a=1 and it doesn't matter
            valid_transform=True,
        )
    }

    return cosmo_args


# These are the same as above, but they are extracted as real python strings
# instead of bytecode characters.
header_unpack_string = {
    "RunName": "run_name",
    "SelectOutput": "select_output",
    "OutputType": "output_type",
    "System": "system_name",
}


# These are the same as above, but they are stored in the snapshots as length
# one arrays. These values will be unpacked, i.e. they will be set as
# self.y = self.header[x][0].
header_unpack_single_float = {
    "Time": ["time", "t"],
    "Dimension": "dimension",
    "Redshift": ["redshift", "z"],
    "Scale-factor": "scale_factor",
}


def generate_units_header_unpack_single_float(m, l, t, I, T) -> dict:
    """
    Generate the unit dictionaries with the:

    mass, length, time, current, and temperature

    units respectively.
    """

    units = {"time": t, "t": t}

    return units
