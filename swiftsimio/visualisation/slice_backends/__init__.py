"""
Backends for density slicing.
"""

from swiftsimio.visualisation.slice_backends.sph import (
    slice_scatter as sph,
    slice_scatter_parallel as sph_parallel,
    get_hsml as sph_get_hsml,
)
from swiftsimio.visualisation.slice_backends.nearest_neighbors import (
    slice_scatter as nearest_neighbors,
    slice_scatter_parallel as nearest_neighbors_parallel,
    get_hsml as nearest_neighbors_get_hsml,
)

backends = {"sph": sph, "nearest_neighbors": nearest_neighbors}

backends_parallel = {
    "sph": sph_parallel,
    "nearest_neighbors": nearest_neighbors_parallel,
}

backends_get_hsml = {
    "sph": sph_get_hsml,
    "nearest_neighbors": nearest_neighbors_get_hsml,
}
