"""Basic integration test."""

import os
from swiftsimio import load


def test_write(simple_snapshot_data):
    """Create a sample dataset. Should not crash."""
    _, testfile = simple_snapshot_data
    assert os.path.isfile(testfile)


def test_load(simple_snapshot_data):
    """Try to load a dataset made by the writer. Should not crash."""
    _, testfile = simple_snapshot_data
    dat = load(testfile)
    dat.gas.internal_energy
    dat.gas.coordinates
