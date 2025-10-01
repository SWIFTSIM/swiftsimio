"""
Tests some known good states with the metadata.
"""

from swiftsimio import metadata, load, SWIFTUnits
from swiftsimio.metadata.objects import SWIFTSnapshotMetadata


def test_same_contents():
    """
    Tests that there are the same arrays in each of the following:

        + particle fields
        + unit fields
        + cosmology fields

    We treat the particle fields as the ground truth.
    """

    cosmology = metadata.cosmology_fields.generate_cosmology(1.0, 1.0)
    units = metadata.unit_fields.generate_units(1.0, 1.0, 1.0, 1.0, 1.0)
    particle = {x: getattr(metadata.particle_fields, x) for x in units.keys()}

    # Do we cover all the same particle fields?

    assert cosmology.keys() == particle.keys()
    assert units.keys() == particle.keys()

    for ptype in cosmology.keys():
        assert list(units[ptype].keys()) == list(particle[ptype].values())
        assert list(cosmology[ptype].keys()) == list(particle[ptype].values())

    return


def test_file_handle_cleanup(cosmological_volume_only_single):
    """
    Check that file handle is cleaned up when no longer needed.

    https://github.com/SWIFTSIM/swiftsimio/pull/155 introduced using a single file
    handle for all metadata reading to prevent too many file requests to file metadata
    servers when applicable (e.g. on cosma). This risks leaving the file handle open,
    which can lead to e.g. being unable to delete a file at the OS level on some
    platforms, like Windows.

    This tests that the file handle is released by the time constructing a SWIFTUnits,
    SWIFTMetadata or SWIFTDataset object is finished.
    """
    units = SWIFTUnits(cosmological_volume_only_single)
    assert not units._handle  # asserts True if file open, False if closed

    metadata = SWIFTSnapshotMetadata(cosmological_volume_only_single)
    assert not metadata.units._handle

    data = load(cosmological_volume_only_single)
    assert not data.metadata.units._handle
