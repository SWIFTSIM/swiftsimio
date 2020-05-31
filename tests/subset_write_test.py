from tests.helper import requires

import numpy as np
from swiftsimio.subset_writer import write_subset, find_datasets
import swiftsimio as sw
import h5py
import sys
import os

import faulthandler
from time import perf_counter

def compare(A, B):
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

    # Compare metadata
    if A.metadata.time != B.metadata.time:
        bad_compares.append("metadata")

    # Compare datasets
    for part_type in filter(
        lambda x: hasattr(A, x),
        sw.metadata.particle_types.particle_name_underscores.values(),
    ):
        A_type = getattr(A, part_type)
        B_type = getattr(B, part_type)
        for attr in filter(lambda x: not x.startswith("_"), dir(A_type)):
            param = getattr(A_type, attr)
            if "coordinates" in attr:
                print(part_type)
                print(param)
                print(getattr(B_type, attr))
            if not callable(param):
                try:
                    if not (param == getattr(B_type, attr)):
                        bad_compares.append(f"{part_type} {attr}")
                        print(param)
                        print(getattr(B_type, attr))
                except:
                    if not (param == getattr(B_type, attr)).all():
                        bad_compares.append(f"{part_type} {attr}")
                        print(param)
                        print(getattr(B_type, attr))

    assert bad_compares == [], f"compare failed on {bad_compares}"


#@requires("cosmological_volume.hdf5")
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
    box_length = boxsize[0]
    n_cells = 96
    half_cell_size = box_length/(n_cells*2)

    # Decide which region we want to load
    load_region = [[0.5 * b - half_cell_size, 0.5 * b + half_cell_size] for b in boxsize]
    mask.constrain_spatial(load_region)

    # Write the subset
    t1 = perf_counter()
    write_subset(outfile, mask)
    t2 = perf_counter()
    print("time elapsed {:.3e}s".format(t2 - t1))

    # Compare written subset of snapshot against corresponding region in full snapshot
    snapshot = sw.load(filename, mask)
    sub_snapshot = sw.load(outfile)

    # First check the metadata
    #compare(snapshot, sub_snapshot)

    # Clean up
    os.remove(outfile)

    return

faulthandler.enable()

#filename = "cosmological_volume.hdf5"
filename = "../subset_writing_speed/eagle_0011.hdf5"

#t1 = perf_counter()
test_subset_writer(filename)
#t2 = perf_counter()
#print("time elapsed {:.3e}s".format(t2 - t1))
