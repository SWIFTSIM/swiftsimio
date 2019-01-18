"""
Contains the description of the metadata fields in the SWIFT snapshots.
"""

metadata_fields_to_read = {
    "Code": "code",
    "Cosmology": "cosmology",
    "Header": "header",
    "HydroScheme": "hydro_scheme",
    "InternalCodeUnits": "internal_code_units",
    "Parameters": "parameters",
    "Policy": "policy",
    "RuntimePars": "runtime_pars",
    "SubgridScheme": "subgrid_scheme",
    "UnusedParameters": "unused_parameters",
}


# These will be unpacked to the top-level object. Be careful not to overwrite
# things in the same namespace!
header_unpack_variables = {
    "BoxSize": "boxsize",
    "NumPart_Total": "num_part",
    "MassTable": "mass_table",
}

# These are the same as above, but they are stored in the snapshots as length
# one arrays. These values will be unpacked, i.e. they will be set as
# self.y = self.header[x][0].
header_unpack_single_float = {
    "Time": "time",
    "Time": "t",
    "Dimension": "dimension",
    "Redshift": "redshift",
    "Redshift": "z",
    "Scale-factor": "scale_factor",
    "Scale-factor": "a",
}
