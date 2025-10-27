"""Tests some known good states with the metadata."""

import numpy as np
import h5py
from swiftsimio import metadata, load, mask, SWIFTUnits, cosmo_array
from swiftsimio.metadata.objects import SWIFTSnapshotMetadata


def _is_closed_hdf5_file(handle: h5py.File) -> bool:
    """
    Check if a hdf5 file handle is closed.

    Parameters
    ----------
    handle : h5py.File
        The handle to check for open state.

    Returns
    -------
    bool
        ``True`` if ``handle`` is a closed file handle, else ``False``.
    """
    return isinstance(handle, h5py.File) and not handle


def test_same_contents():
    """
    Tests that field metadata defined by hand covers same fields.

    Checks that there are the same arrays in each of the following:

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
        assert set(units[ptype].keys()) == set(particle[ptype].values())
        assert set(cosmology[ptype].keys()) == set(particle[ptype].values())

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
    assert _is_closed_hdf5_file(
        units._handle
    )  # asserts True if file open, False if closed

    metadata = SWIFTSnapshotMetadata(cosmological_volume_only_single)
    assert _is_closed_hdf5_file(metadata._handle)
    assert _is_closed_hdf5_file(metadata.units._handle)

    m = mask(cosmological_volume_only_single)
    assert _is_closed_hdf5_file(m._handle)
    assert _is_closed_hdf5_file(m.metadata._handle)
    assert _is_closed_hdf5_file(m.units._handle)
    assert _is_closed_hdf5_file(m.metadata.units._handle)

    data = load(cosmological_volume_only_single)
    assert _is_closed_hdf5_file(data._handle)
    assert _is_closed_hdf5_file(data.metadata._handle)
    assert _is_closed_hdf5_file(data.units._handle)
    assert _is_closed_hdf5_file(data.gas._handle)
    assert _is_closed_hdf5_file(data.gas.metadata._handle)
    assert _is_closed_hdf5_file(data.metadata.units._handle)
    assert _is_closed_hdf5_file(data.gas.metadata.units._handle)

    m.constrain_spatial(
        cosmo_array([np.zeros_like(m.metadata.boxsize), m.metadata.boxsize]).T
    )
    mdata = load(cosmological_volume_only_single, mask=m)
    assert _is_closed_hdf5_file(mdata._handle)
    assert _is_closed_hdf5_file(mdata.metadata._handle)
    assert _is_closed_hdf5_file(mdata.units._handle)
    assert _is_closed_hdf5_file(mdata.gas._handle)
    assert _is_closed_hdf5_file(mdata.gas.metadata._handle)
    assert _is_closed_hdf5_file(mdata.metadata.units._handle)
    assert _is_closed_hdf5_file(mdata.gas.metadata.units._handle)


def test_mask_and_dataset_share_metadata(cosmological_volume_only_single):
    """
    When a mask is used, chec that we skip re-reading the metadata.

    We should not read it in the SWIFTDataset and borrow it from the mask instead.
    Check that this is the case.
    """
    m = mask(cosmological_volume_only_single)
    region = cosmo_array([np.zeros_like(m.metadata.boxsize), m.metadata.boxsize]).T
    m.constrain_spatial(region)
    data = load(cosmological_volume_only_single, mask=m)
    assert data.metadata is m.metadata
    # check that this wasn't trivial for good measure
    unmasked_data = load(cosmological_volume_only_single)
    assert unmasked_data.metadata is not m.metadata


def test_file_handle_shared(cosmological_volume_only_single):
    """
    Check that file handles are shared across objects.

    Only the handle for the initial burst of metadata reading is shared. Later reading
    of datasets and their attributes get a new handle internally.
    """
    data = load(cosmological_volume_only_single)
    assert data.metadata._handle is data._handle
    assert data.units._handle is data._handle
    assert data.metadata.units._handle is data._handle
    assert data.gas._handle is data._handle
    assert data.gas.metadata._handle is data._handle


def test_file_handle_shared_when_masked(cosmological_volume_only_single):
    """
    Check that file handles are shared across objects when a mask is used.

    See also ``test_file_handle_shared`` for the unmasked case.

    In the masked case we have a different expectation because the mask is created
    separately and therefore has its own file handles. The dataset, when created,
    doesn't re-read the metadata, but borrows it from the mask (see
    ``test_mask_and_dataset_share_metadata``) so we don't expect the metadata
    to have the same handle in this case.
    """
    m = mask(cosmological_volume_only_single)  # gets, uses and closes its own handle
    region = cosmo_array([np.zeros_like(m.metadata.boxsize), m.metadata.boxsize]).T
    m.constrain_spatial(region)
    data = load(cosmological_volume_only_single, mask=m)
    # comparison with `is` robust to handle being open/closed
    assert data._handle is data._handle
    assert data.metadata._handle is not data._handle  # handle from mask
    assert data.units._handle is not data._handle  # handle from masks
    assert data.metadata.units._handle is not data._handle  # handle from mask
    assert data.gas._handle is data._handle
    assert data.gas.metadata._handle is not data._handle  # handle from mask
    # check that mask shared its handle:
    assert data.units._handle is data.metadata._handle
    assert data.metadata.units._handle is data.metadata._handle
    assert data.gas.metadata._handle is data.metadata._handle
