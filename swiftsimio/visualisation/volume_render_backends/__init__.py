"""
Backends for volume rendering
"""

from swiftsimio.visualisation.volume_render_backends.scatter import (
    scatter,
    scatter_parallel,
)

backends = {"scatter": scatter}

backends_parallel = {"scatter": scatter_parallel}
