"""
Handle metadata for data files.

Metadata includes a listing of the datasets available in a file.
"""

from .particle import particle_types, particle_fields
from .soap import soap_types
from .unit import unit_types, unit_fields
from .metadata import metadata_fields
from .writer import required_fields

__all__ = [
    "particle_types",
    "particle_fields",
    "soap_types",
    "unit_types",
    "unit_fields",
    "metadata_fields",
    "required_fields",
]
