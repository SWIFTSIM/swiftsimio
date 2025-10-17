"""
Contains the metadata for all of the snapshots, such as the
information contained in each snapshot that may be read.
"""

from .particle import particle_types, particle_fields
from .soap import soap_types
from .unit import unit_types, unit_fields
from .metadata import metadata_fields
from .cosmology import cosmology_fields
from .writer import required_fields

__all__ = [
    "particle_types",
    "particle_fields",
    "soap_types",
    "unit_types",
    "unit_fields",
    "metadata_fields",
    "cosmology_fields",
    "required_fields",
]
