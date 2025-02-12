"""
Backends for ray tracing
"""

from swiftsimio.visualisation._vistools import backends_restore_cosmo_and_units

from swiftsimio.visualisation.ray_trace_backends.core_panels import (
    core_panels,
    core_panels_parallel,
)

backends = {"core_panels": backends_restore_cosmo_and_units(core_panels)}

backends_parallel = {
    "core_panels": backends_restore_cosmo_and_units(core_panels_parallel)
}
