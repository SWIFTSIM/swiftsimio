import numpy as np
from swiftsimio.subset_writer import write_subset
import swiftsimio as sw
import h5py
import sys

def compare(A, B):
    # ALEXEI: don't be so stringent on comparing all elements of metadata, focus on dataset instead
    val = True
    for key in A.__dict__.keys():
        print("comparing ", key)
        try:
            if A.__dict__[key] != B.__dict__[key] :
                val = False
        except TypeError:
            if not all(A.__dict__[key][:] == B.__dict__[key][:]):
                val = False
        except:
            print(A.__dict__[key])

    if val:
        print("everything's fine")
    else:
        print("something's wrong")
        sys.exit()


# Specify filepaths
infile = "/cosma7/data/dp004/dc-bori1/swiftsimio_project/snapshots/test_snap_eagle25.hdf5"
outfile = "/cosma7/data/dp004/dc-bori1/swiftsimio_project/snapshots/test_out.hdf5"

# Create a mask
mask = sw.mask(infile)

boxsize = mask.metadata.boxsize

# Decide which region we want to load
load_region = [[0.49 * b, 0.51*b] for b in boxsize]
mask.constrain_spatial(load_region)

mask_size = np.asarray(mask).size
mask_size = np.sum(mask.gas[:,1]) - np.sum(mask.gas[:,0])

# Write the subset
write_subset(infile, outfile, mask)

## Compare written subset of snapshot against corresponding region in full snapshot
#snapshot = sw.load(infile)
#sub_snapshot = sw.load(outfile)
#
## First check the metadata
#compare(snapshot.metadata, sub_snapshot.metadata)

