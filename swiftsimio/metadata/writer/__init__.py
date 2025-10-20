"""A list of required fields for writing a SWIFT dataset, for each particle type."""

from .required_fields import (
    gas,
    dark_matter,
    boundary,
    sinks,
    stars,
    black_holes,
    neutrinos,
)

__all__ = [
    "gas",
    "dark_matter",
    "boundary",
    "sinks",
    "stars",
    "black_holes",
    "neutrinos",
]
