"""
Backends for density projection.

These go in order (within the dictionary) from
fastest to most accurate, with the "_reference" style
being a developer-only indended feature.
"""
from numba.cuda.cudadrv.error import CudaSupportError

from swiftsimio.visualisation.projection_backends.fast import scatter as fast
from swiftsimio.visualisation.projection_backends.fast import (
    scatter_parallel as fast_parallel,
)

from swiftsimio.visualisation.projection_backends.histogram import scatter as histogram
from swiftsimio.visualisation.projection_backends.histogram import (
    scatter_parallel as histogram_parallel,
)

from swiftsimio.visualisation.projection_backends.reference import scatter as reference
from swiftsimio.visualisation.projection_backends.reference import (
    scatter_parallel as reference_parallel,
)

from swiftsimio.visualisation.projection_backends.renormalised import (
    scatter as renormalised,
)
from swiftsimio.visualisation.projection_backends.renormalised import (
    scatter_parallel as renormalised_parallel,
)

from swiftsimio.visualisation.projection_backends.subsampled import (
    scatter as subsampled,
)
from swiftsimio.visualisation.projection_backends.subsampled import (
    scatter_parallel as subsampled_parallel,
)

from swiftsimio.visualisation.projection_backends.subsampled_extreme import (
    scatter as subsampled_extreme,
)
from swiftsimio.visualisation.projection_backends.subsampled_extreme import (
    scatter_parallel as subsampled_extreme_parallel,
)

backends = {
    "histogram": histogram,
    "fast": fast,
    "renormalised": renormalised,
    "subsampled": subsampled,
    "subsampled_extreme": subsampled_extreme,
    "reference": reference,
}

backends_parallel = {
    "histogram": histogram_parallel,
    "fast": fast_parallel,
    "renormalised": renormalised_parallel,
    "subsampled": subsampled_parallel,
    "subsampled_extreme": subsampled_extreme_parallel,
    "reference": reference_parallel,
}

try:
    from swiftsimio.visualisation.projection_backends.gpu import scatter \
        as scatter_gpu
    from swiftsimio.visualisation.projection_backends.gpu import (
        scatter_parallel as scatter_gpu_parallel,
    )

    backends["gpu"] = scatter_gpu
    backends_parallel["gpu"] = scatter_gpu_parallel
except CudaSupportError:
    print(
        "Unable to load the GPU module. Please check the module numba.cuda "
        "if you wish to use them."
    )
