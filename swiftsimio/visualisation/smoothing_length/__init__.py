"""Tools for smoothing length generation."""

import numpy as np
from .sph import get_hsml as _sph_get_hsml
from .nearest_neighbours import get_hsml as _nearest_neighbours_get_hsml
from .generate import generate_smoothing_lengths

__all__ = [
    "generate_smoothing_lengths",
    "backends_get_hsml",
]

backends_get_hsml = {
    "histogram": lambda data: np.empty_like(data.masses),
    "sph": _sph_get_hsml,
    "nearest_neighbours": _nearest_neighbours_get_hsml,
}
