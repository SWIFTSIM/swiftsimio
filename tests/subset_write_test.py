from tests.helper import requires

import numpy as np
from swiftsimio.subset_writer import write_subset, find_datasets
import swiftsimio as sw
import h5py
import sys
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

    # Compare metadata - this is non-trivial so we just compare the time as a
    # sanity check.
    if A.metadata.time != B.metadata.time:
        bad_compares.append("metadata")

    # Compare datasets
    for part_type in filter(
        lambda x: hasattr(A, x),
        sw.metadata.particle_types.particle_name_underscores.values(),
    ):
        A_type = getattr(A, part_type)
        B_type = getattr(B, part_type)
        particle_dataset_field_names = set(
            A_type.particle_metadata.field_names + B_type.particle_metadata.field_names
        )

        for attr in particle_dataset_field_names:
            param_A = getattr(A_type, attr)
            param_B = getattr(B_type, attr)
            try:
                if not (param_A == param_B):
                    bad_compares.append(f"{part_type} {attr}")
            except:
                if not (param_A == param_B).all():
                    bad_compares.append(f"{part_type} {attr}")

    assert bad_compares == [], f"compare failed on {bad_compares}"


@requires("cosmological_volume.hdf5")
def test_subset_writer(filename):
    """
    Test to make sure subset writing works as intended

    Writes a subset of the cosmological volume to a snapshot file
    and compares result against masked load of the original file.
    """
    # Specify output filepath
    outfile = "subset_cosmological_volume.hdf5"

    # Create a mask
    mask = sw.mask(filename)

    boxsize = mask.metadata.boxsize

    # Decide which region we want to load
    load_region = [[0.25 * b, 0.75 * b] for b in boxsize]
    mask.constrain_spatial(load_region)

    # Write the subset
    write_subset(outfile, mask)

    # Compare subset of written subset of snapshot against corresponding region in
    # full snapshot. This checks that both the metadata and dataset subsets are
    # written properly.
    sub_mask = sw.mask(outfile)
    sub_load_region = [[0.375 * b, 0.625 * b] for b in boxsize]
    sub_mask.constrain_spatial(sub_load_region)
    mask.constrain_spatial(sub_load_region)

    snapshot = sw.load(filename, mask)
    sub_snapshot = sw.load(outfile, sub_mask)

    compare_data_contents(snapshot, sub_snapshot)

    # Clean up
    os.remove(outfile)

    return
