import numpy as np
from .sph import get_hsml as sph_get_hsml
from .nearest_neighbours import get_hsml as nearest_neighbours_get_hsml
from .generate import generate_smoothing_lengths

backends_get_hsml = {
    "histogram": lambda m: np.empty_like(m),
    "sph": sph_get_hsml,
    "nearest_neighbours": nearest_neighbours_get_hsml,
}
