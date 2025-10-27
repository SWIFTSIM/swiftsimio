"""Backends for density slicing."""

from swiftsimio.visualisation.slice_backends.sph import (
    slice_scatter as sph,
    slice_scatter_parallel as sph_parallel,
)
from swiftsimio.visualisation.slice_backends.nearest_neighbours import (
    slice_scatter as nearest_neighbours,
    slice_scatter_parallel as nearest_neighbours_parallel,
)

backends = {"sph": sph, "nearest_neighbours": nearest_neighbours}

backends_parallel = {
    "sph": sph_parallel,
    "nearest_neighbours": nearest_neighbours_parallel,
}
