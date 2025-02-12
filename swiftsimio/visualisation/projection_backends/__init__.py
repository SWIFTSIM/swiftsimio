"""
Backends for density projection.

These go in order (within the dictionary) from
fastest to most accurate, with the "_reference" style
being a developer-only indended feature.
"""

from swiftsimio.visualisation._vistools import backends_restore_cosmo_and_units

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

from swiftsimio.visualisation.projection_backends.gpu import scatter as gpu
from swiftsimio.visualisation.projection_backends.gpu import (
    scatter_parallel as gpu_parallel,
)

backends = {
    "histogram": backends_restore_cosmo_and_units(histogram),
    "fast": backends_restore_cosmo_and_units(fast),
    "renormalised": backends_restore_cosmo_and_units(renormalised),
    "subsampled": backends_restore_cosmo_and_units(subsampled),
    "subsampled_extreme": backends_restore_cosmo_and_units(subsampled_extreme),
    "reference": backends_restore_cosmo_and_units(reference),
    "gpu": backends_restore_cosmo_and_units(gpu),
}

backends_parallel = {
    "histogram": backends_restore_cosmo_and_units(histogram_parallel),
    "fast": backends_restore_cosmo_and_units(fast_parallel),
    "renormalised": backends_restore_cosmo_and_units(renormalised_parallel),
    "subsampled": backends_restore_cosmo_and_units(subsampled_parallel),
    "subsampled_extreme": backends_restore_cosmo_and_units(subsampled_extreme_parallel),
    "reference": backends_restore_cosmo_and_units(reference_parallel),
    "gpu": backends_restore_cosmo_and_units(gpu_parallel),
}
