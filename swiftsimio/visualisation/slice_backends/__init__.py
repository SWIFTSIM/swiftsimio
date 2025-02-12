"""
Backends for density slicing.
"""

from swiftsimio.visualisation._vistools import backends_restore_cosmo_and_units

from swiftsimio.visualisation.slice_backends.sph import (
    slice_scatter as sph,
    slice_scatter_parallel as sph_parallel,
)
from swiftsimio.visualisation.slice_backends.nearest_neighbours import (
    slice_scatter as nearest_neighbours,
    slice_scatter_parallel as nearest_neighbours_parallel,
)

backends = {
    "sph": backends_restore_cosmo_and_units(sph),
    "nearest_neighbours": backends_restore_cosmo_and_units(nearest_neighbours),
}

backends_parallel = {
    "sph": backends_restore_cosmo_and_units(sph_parallel),
    "nearest_neighbours": backends_restore_cosmo_and_units(nearest_neighbours_parallel),
}
