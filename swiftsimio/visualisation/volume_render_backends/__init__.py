"""
Backends for volume rendering
"""

from swiftsimio.visualisation._vistools import backends_restore_cosmo_and_units

from swiftsimio.visualisation.volume_render_backends.scatter import (
    scatter,
    scatter_parallel,
)

backends = {"scatter": backends_restore_cosmo_and_units(scatter)}

backends_parallel = {"scatter": backends_restore_cosmo_and_units(scatter_parallel)}
