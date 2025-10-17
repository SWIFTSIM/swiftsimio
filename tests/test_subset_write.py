from swiftsimio.subset_writer import write_subset
from swiftsimio import load
from .helper import _mask_without_warning as mask
import os


def compare_data_contents(A, B):
    """
    Compares two SWIFTDatasets

    Compares all datasets, and one of the metadata fields (this is
    because all metadata fields are copied over simultaneously so if
    they should be either all right or all wrong)

    Parameters
    ----------
    A, B : SWIFTDataset
        datasets to compare
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
    Test to make sure subset writing works as intended

    Writes a subset of the input file to a new file
    and compares result against masked load of the original file.
    """
    # Specify output filepath
    outfile = snapshot_or_soap.replace(".hdf5", "_subset.hdf5")

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
    # Update the spatial region to match what we load from the subset.
    full_mask.constrain_spatial(sub_load_region)

    snapshot = load(snapshot_or_soap, full_mask)
    sub_snapshot = load(outfile, sub_mask)

    compare_data_contents(snapshot, sub_snapshot)

    # Clean up
    os.remove(outfile)

    return
