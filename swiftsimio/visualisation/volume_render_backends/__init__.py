"""Backends for volume rendering."""

from swiftsimio.visualisation.volume_render_backends.scatter import (
    scatter,
    scatter_parallel,
)
from swiftsimio.visualisation.volume_render_backends.nested_grids_scatter import (
    scatter as nested,
    scatter_parallel as nested_parallel,
)

backends = {"scatter": scatter, "nested": nested}

backends_parallel = {"scatter": scatter_parallel, "nested": nested_parallel}
