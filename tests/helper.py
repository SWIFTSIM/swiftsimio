"""
Contains helper functions for the test routines.
"""

import subprocess
import os
import h5py
from swiftsimio.subset_writer import find_links, write_metadata
from swiftsimio import mask, cosmo_array, load
from numpy import mean

webstorage_location = "http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/"
test_data_location = "test_data/"


def requires(filename):
    """
    Use this as a decorator around tests that require data.
    """

    # First check if the test data directory exists
    if not os.path.exists(test_data_location):
        os.mkdir(test_data_location)

    file_location = f"{test_data_location}{filename}"

    if os.path.exists(file_location):
        ret = 0
    else:
        # Download it!
        ret = subprocess.call(
            ["wget", f"{webstorage_location}{filename}", "-O", file_location]
        )

    if ret != 0:
        Warning(f"Unable to download file at {filename}")
        # It wrote an empty file, kill it.
        subprocess.call(["rm", file_location])

        def dont_call_test(func):
            def empty(*args, **kwargs):
                return True

            return empty

        return dont_call_test

    else:
        # Woo, we can do the test!

        def do_call_test(func):
            def final_test():
                # Whack the path on there for good measure.
                return func(f"{test_data_location}{filename}")

            return final_test

        return do_call_test

    raise Exception("You should never have got here.")


def create_in_memory_hdf5(filename="f1"):
    """
    Creates an in-memory hdf5 file object.
    """

    return h5py.File(filename, driver="core", mode="a", backing_store=False)

def create_single_particle_dataset(filename: str, output_name: str):
    """
    Create an hdf5 snapshot with two particles at an identical location

    Parameters
    ----------
    filename: str
        name of file from which to copy metadata
    output_name: str
        name of single particle snapshot
    """
    # Create a dummy mask in order to write metadata
    data_mask = mask(filename)
    boxsize = data_mask.metadata.boxsize
    region = [[0, b] for b in boxsize]
    data_mask.constrain_spatial(region)

    # Write the metadata
    infile = h5py.File(filename, "r")
    outfile = h5py.File(output_name, "w")
    list_of_links, _ = find_links(infile)
    write_metadata(infile, outfile, list_of_links, data_mask)

    # Write a single particle
    particle_coords = cosmo_array([[1,1,1], [1,1,1]], data_mask.metadata.units.length)
    particle_masses = cosmo_array([1, 1], data_mask.metadata.units.mass)
    mean_h = mean(infile["/PartType0/SmoothingLengths"])
    particle_h = cosmo_array([mean_h, mean_h], data_mask.metadata.units.length)
    particle_ids = [1, 2]

    coords = outfile.create_dataset("/PartType0/Coordinates", data=particle_coords)
    for name, value in infile["/PartType0/Coordinates"].attrs.items():
        coords.attrs.create(name, value)

    masses = outfile.create_dataset("/PartType0/Masses", data=particle_masses)
    for name, value in infile["/PartType0/Masses"].attrs.items():
        masses.attrs.create(name, value)

    h = outfile.create_dataset("/PartType0/SmoothingLengths", data=particle_h)
    for name, value in infile["/PartType0/SmoothingLengths"].attrs.items():
        h.attrs.create(name, value)

    ids = outfile.create_dataset("/PartType0/ParticleIDs", data=particle_ids)
    for name, value in infile["/PartType0/ParticleIDs"].attrs.items():
        ids.attrs.create(name, value)

    # Get rid of all traces of DM
    del outfile["/Cells/Counts/PartType1"]
    del outfile["/Cells/Offsets/PartType1"]
    nparts_total = [2, 0, 0, 0, 0, 0]
    nparts_this_file = [2, 0, 0, 0, 0, 0]
    outfile["/Header"].attrs["NumPart_Total"] = nparts_total
    outfile["/Header"].attrs["NumPart_ThisFile"] = nparts_this_file

    # Tidy up
    infile.close()
    outfile.close()

    return

