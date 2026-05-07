"""Tests of the subset writer feature."""

import pytest
import numpy as np
from swiftsimio.subset_writer import write_subset
from swiftsimio import load, SWIFTDataset
from .helper import _mask_without_warning as mask
import os
import os.path
from pathlib import Path


def compare_data_contents(A: SWIFTDataset, B: SWIFTDataset) -> None:
    """
    Compare two SWIFTDatasets.

    Compares all datasets, and one of the metadata fields (this is
    because all metadata fields are copied over simultaneously so if
    they should be either all right or all wrong)

    Parameters
    ----------
    A : SWIFTDataset
        First dataset to compare.

    B : SWIFTDataset
        Second dataset to compare.

    Raises
    ------
    AssertionError
        If the two datasets are not equivalent.
    """
    # Initialise a list to store fields that differ
    bad_compares = []

    # Compare metadata - this is non-trivial so we just compare the run name as a
    # sanity check.
    if A.metadata.run_name != B.metadata.run_name:
        bad_compares.append("metadata")

    # Compare datasets
    test_was_trivial = True  # make sure we match at least some non-empty arrays
    for group_name in A.metadata.present_group_names:
        A_type = getattr(A, group_name)
        B_type = getattr(B, group_name)
        dataset_field_names = set(
            A_type.group_metadata.field_names + B_type.group_metadata.field_names
        )

        for attr in dataset_field_names:
            param_A = getattr(A_type, attr)
            param_B = getattr(B_type, attr)
            if len(param_A) == 0 and len(param_B) == 0:
                # both arrays are empty, counts as a match
                continue
            else:
                # compared at least one non-empty data array
                test_was_trivial = False
            comparison = param_A == param_B
            if type(comparison) is bool:  # guards len in elif, don't merge nested if
                if not comparison:
                    bad_compares.append(f"{group_name} {attr}")
            elif len(comparison) > 1:
                if not comparison.all():
                    bad_compares.append(f"{group_name} {attr}")

    assert bad_compares == [], f"compare failed on {bad_compares}"
    assert not test_was_trivial


def test_subset_writer(snapshot_or_soap):
    """
    Test to make sure subset writing works as intended.

    Writes a subset of the input file to a new file
    and compares result against masked load of the original file.
    """
    # Get the name of the input test file
    if isinstance(snapshot_or_soap, (Path, str)):
        filename = str(snapshot_or_soap)
    else:
        filename = snapshot_or_soap.filename

    # Specify output filepath
    outfile = os.path.basename(filename).replace(".hdf5", "_subset.hdf5")

    # Create a mask
    full_mask = mask(snapshot_or_soap)
    load_region = [[0.25 * b, 0.75 * b] for b in full_mask.metadata.boxsize]
    full_mask.constrain_spatial(load_region)

    # Write the subset
    write_subset(outfile, full_mask)

    # Compare subset of written subset of snapshot against corresponding region in
    # full snapshot. This checks that both the metadata and dataset subsets are
    # written properly.
    sub_mask = mask(outfile)
    sub_load_region = [[0.375 * b, 0.625 * b] for b in sub_mask.metadata.boxsize]
    sub_mask.constrain_spatial(sub_load_region)
    # Match what we load from the subset for the full snapshot.
    full_mask_small = mask(snapshot_or_soap)
    full_mask_small.constrain_spatial(sub_load_region)

    snapshot = load(snapshot_or_soap, full_mask_small)
    sub_snapshot = load(outfile, sub_mask)

    compare_data_contents(snapshot, sub_snapshot)

    # Clean up
    os.remove(outfile)

    return


@pytest.mark.parametrize("range_mask", (True, False))
def test_subset_writer_constrained_indices(soap_example, range_mask):
    """Test that a subset written with constrain_indices has valid metadata."""
    filename = (
        str(soap_example)
        if isinstance(soap_example, (Path, str))
        else soap_example.filename
    )
    m = mask(soap_example, range_mask=range_mask)
    region = np.vstack([m.metadata.boxsize * 0, m.metadata.boxsize * 0.5]).T
    m.constrain_indices([1, 2, 3])
    outfile = os.path.basename(filename).replace(".hdf5", "_subset.hdf5")
    write_subset(outfile, m)
    # try loading with a spatial mask to make sure cell metadata is ok:
    sub_mask = mask(outfile)
    sub_mask.constrain_spatial(region)
    sub_dat = load(outfile, mask=sub_mask)
    sub_dat.bound_subhalo.total_mass
    # clean up
    os.remove(outfile)


def test_masking_subset(snapshot_or_soap):
    """
    Test that we can select a sub-region of a subset written to file.

    We write out an octant of a snapshot or soap catalogue as a new file, then load a
    sub-region of that octant from both the full file and the file with just the octant.
    Finally we compare the contents of those two masked datasets to make sure that they
    match.
    """
    filename = (
        str(snapshot_or_soap)
        if isinstance(snapshot_or_soap, (Path, str))
        else snapshot_or_soap.filename
    )
    octant_mask = mask(snapshot_or_soap)
    boxsize = octant_mask.metadata.boxsize
    octant_region = np.vstack([boxsize * 0.5, boxsize]).T
    octant_mask.constrain_spatial(octant_region)
    outfile = os.path.basename(filename).replace(".hdf5", "_octant.hdf5")
    write_subset(outfile, octant_mask)
    # have to be a bit careful to pick a region with at least a subhalo in it for soap,
    # otherwise it's a trivial comparison and the test fails:
    small_region = np.vstack([boxsize * 0.8, boxsize * 0.8001]).T
    small_mask_full = mask(snapshot_or_soap)
    small_mask_sub = mask(outfile)
    small_mask_full.constrain_spatial(small_region)
    small_mask_sub.constrain_spatial(small_region)
    d_full = load(snapshot_or_soap, mask=small_mask_full)
    d_sub = load(outfile, mask=small_mask_sub)
    compare_data_contents(d_full, d_sub)
    # clean up
    os.remove(outfile)
