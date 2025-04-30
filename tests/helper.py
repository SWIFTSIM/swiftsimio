"""
Contains helper functions for the test routines.
"""

import h5py
from swiftsimio.subset_writer import find_links, write_metadata
from swiftsimio import mask, cosmo_array
from numpy import mean, zeros_like


def create_in_memory_hdf5(filename="f1"):
    """
    Creates an in-memory hdf5 file object.
    """

    return h5py.File(filename, driver="core", mode="a", backing_store=False)


def create_n_particle_dataset(filename: str, output_name: str, num_parts: int = 2):
    """
    Create an hdf5 snapshot with a desired number of identical particles.

    Parameters
    ----------
    filename: str
        name of file from which to copy metadata
    output_name: str
        name of single particle snapshot
    num_parts: int
        number of particles to create (default: 2)
    """
    # Create a dummy mask in order to write metadata
    data_mask = mask(filename)
    boxsize = data_mask.metadata.boxsize
    region = [[zeros_like(b), b] for b in boxsize]
    data_mask.constrain_spatial(region)

    # Write the metadata
    infile = h5py.File(filename, "r")
    outfile = h5py.File(output_name, "w")
    list_of_links, _ = find_links(infile)
    write_metadata(infile, outfile, list_of_links, data_mask)

    # Write a single particle
    particle_coords = cosmo_array(
        [[1, 1, 1]] * num_parts, data_mask.metadata.units.length
    )
    particle_masses = cosmo_array([1] * num_parts, data_mask.metadata.units.mass)
    mean_h = mean(infile["/PartType0/SmoothingLengths"])
    particle_h = cosmo_array([mean_h, mean_h], data_mask.metadata.units.length)
    particle_ids = list(range(1, num_parts + 1))

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
    nparts_total = [num_parts, 0, 0, 0, 0, 0]
    nparts_this_file = [num_parts, 0, 0, 0, 0, 0]
    outfile["/Header"].attrs["NumPart_Total"] = nparts_total
    outfile["/Header"].attrs["NumPart_ThisFile"] = nparts_this_file

    # Tidy up
    infile.close()
    outfile.close()

    return
