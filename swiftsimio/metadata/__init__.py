"""
Handle metadata for data files.

Metadata includes a listing of the datasets available in a file.
"""

from .particle import particle_types
from .soap import soap_types
from .unit import unit_types
from .metadata import metadata_fields
from .writer import required_fields

__all__ = [
    "particle_types",
    "soap_types",
    "unit_types",
    "metadata_fields",
    "required_fields",
]
