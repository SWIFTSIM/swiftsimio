"""
Contains the description of the metadata fields in the SWIFT snapshots.
"""

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
