from swiftsimio.visualisation.slice_backends.sph import (
    slice_scatter as sph,
    slice_scatter_parallel as sph_parallel,
)
from swiftsimio.visualisation.slice_backends.nearest_neighbors import (
    slice_scatter as nearest_neighbors,
    slice_scatter_parallel as nearest_neighbors_parallel,
)

backends = {
    "sph": sph,
    "nearest_neighbours": nearest_neighbors,
}

backends_parallel = {
    "sph": sph_parallel,
    "nearest_neighbours": nearest_neighbors_parallel,
}
