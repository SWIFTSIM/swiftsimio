from tests.helper import requires

import numpy as np
from swiftsimio.subset_writer import write_subset, find_datasets
import swiftsimio as sw
import h5py
import sys

def compare_arrays(A, B):
    # Check we're not going crazy
    if len(A) != len(B):
        return False
    
    # Compare element-by-element
    for i in range(len(A)):
        try:
            if A[i] != B[i]:
                return False
        except:
            if not all(A[i][:] == B[i][:]):
                return False

    # All good
    return True

def compare(A, B):
    # Initialise a list to store fields that differ
    bad_compares = []

    # Compare metadata
    if A.metadata.time != B.metadata.time:
        bad_compares.append("metadata")

    # Compare datasets
    possible_part_types = ['gas', 'dark_matter', 'stars', 'black_holes']
    part_types = [attr for attr in possible_part_types if hasattr(A, attr)]
    for j in range(len(part_types)):
        A_type = getattr(A, part_types[j])
        B_type = getattr(B, part_types[j])
        attrs = [attr for attr in dir(A_type) if not attr.startswith('_')]
        for i in range(len(attrs)):
            if not callable(getattr(A_type, attrs[i])):
                print("comparing ", attrs[i])
                if not compare_arrays(getattr(A_type, attrs[i]), getattr(B_type, attrs[i])):
                    bad_compares.append(part_types[j] + " " + attrs[i])

    if bad_compares != []:
        print("compare failed on ", bad_compares)
    else:
        print("compare completed successfully")

    assert(bad_compares == [])

@requires("cosmological_volume.hdf5")
def test_subset_writer(filename):
    # Specify output filepath
    outfile = "subset_cosmological_volume.hdf5"
    
    # Create a mask
    mask = sw.mask(filename)
    
    boxsize = mask.metadata.boxsize
    
    # Decide which region we want to load
    load_region = [[0.49 * b, 0.51*b] for b in boxsize]
    mask.constrain_spatial(load_region)
    
    # Write the subset
    write_subset(filename, outfile, mask)
    
    # Compare written subset of snapshot against corresponding region in full snapshot
    snapshot = sw.load(filename, mask)
    sub_snapshot = sw.load(outfile)
    
    # First check the metadata
    compare(snapshot, sub_snapshot)

    return

