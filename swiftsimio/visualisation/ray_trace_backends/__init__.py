"""
Backends for ray tracing
"""

from swiftsimio.visualisation.ray_trace_backends.core_panels import (
    core_panels,
    core_panels_parallel,
)

backends = {"core_panels": core_panels}

backends_parallel = {"core_panels": core_panels_parallel}
