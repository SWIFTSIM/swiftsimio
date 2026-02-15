"""Test loading of IC-like data files created with swiftsimio."""

from swiftsimio import load
from swiftsimio import cosmo_array

import numpy as np


def test_reading_ic_units(simple_snapshot_data):
    """Test to ensure we are able to correctly read ICs created with swiftsimio."""
    writer_instance, test_filename = simple_snapshot_data
    data = load(test_filename)

    # np.allclose checks unit consistency
    for field in [
        "coordinates",
        "velocities",
        "masses",
        "internal_energy",
        "smoothing_lengths",
    ]:
        assert isinstance(getattr(data.gas, field), cosmo_array)
        assert np.allclose(
            getattr(data.gas, field), getattr(writer_instance.gas, field), rtol=1.0e-4
        )
    return
