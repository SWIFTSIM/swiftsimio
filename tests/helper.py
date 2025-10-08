"""
Contains helper functions for the test routines.
"""

import pytest
import h5py
import unyt as u
from swiftsimio.subset_writer import find_links, write_metadata
from swiftsimio import mask, cosmo_array
from numpy import mean, zeros_like


def _mask_without_warning(fname, **kwargs):
    with h5py.File(fname, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
        is_soap = f["/Header"].attrs.get("OutputType", "FullVolume") == "SOAP"
    if has_cell_bbox or is_soap:
        return mask(fname, **kwargs)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            return mask(fname, **kwargs)


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
    data_mask = _mask_without_warning(filename)
    boxsize = data_mask.metadata.boxsize
    region = [[zeros_like(b), b] for b in boxsize]
    data_mask.constrain_spatial(region)

    # Write the metadata
    infile = h5py.File(filename, "r")
    outfile = h5py.File(output_name, "w")
    list_of_links, _ = find_links(infile)
    write_metadata(infile, outfile, list_of_links, data_mask)

    # Write copied particles
    particle_coords = cosmo_array(
        [[1, 1, 1]] * num_parts, data_mask.metadata.units.length
    )
    particle_masses = cosmo_array([1] * num_parts, data_mask.metadata.units.mass)
    mean_h = mean(infile["/PartType0/SmoothingLengths"])
    particle_h = cosmo_array([mean_h] * num_parts, data_mask.metadata.units.length)
    particle_ids = list(range(1, num_parts + 1))
    particle_element_mass_fractions = cosmo_array(
        [1] * num_parts * 9, u.dimensionless
    ).reshape((num_parts, 9))

    coords = outfile.create_dataset(
        "/PartType0/Coordinates", data=particle_coords, shape=(num_parts, 3)
    )
    for name, value in infile["/PartType0/Coordinates"].attrs.items():
        coords.attrs.create(name, value)

    masses = outfile.create_dataset(
        "/PartType0/Masses", data=particle_masses, shape=(num_parts,)
    )
    for name, value in infile["/PartType0/Masses"].attrs.items():
        masses.attrs.create(name, value)

    h = outfile.create_dataset(
        "/PartType0/SmoothingLengths", data=particle_h, shape=(num_parts,)
    )
    for name, value in infile["/PartType0/SmoothingLengths"].attrs.items():
        h.attrs.create(name, value)

    ids = outfile.create_dataset(
        "/PartType0/ParticleIDs", data=particle_ids, shape=(num_parts,)
    )
    for name, value in infile["/PartType0/ParticleIDs"].attrs.items():
        ids.attrs.create(name, value)

    element_mass_fractions = outfile.create_dataset(
        "/PartType0/ElementMassFractions",
        data=particle_element_mass_fractions,
        shape=(num_parts, 9),
    )
    for name, value in infile["/PartType0/ElementMassFractions"].attrs.items():
        element_mass_fractions.attrs.create(name, value)

    # Get rid of all traces of DM
    outfile["/Cells/Counts/PartType0"][...] = 0
    del outfile["/Cells/Counts/PartType1"]
    if "Offsets" in outfile["/Cells"].keys():
        outfile["/Cells/Offsets/PartType0"][...] = 0
        del outfile["/Cells/Offsets/PartType1"]
    if "OffsetsInFile" in outfile["/Cells"].keys():
        outfile["/Cells/OffsetsInFile/PartType0"][...] = 0
        del outfile["/Cells/OffsetsInFile/PartType1"]
    nparts_total = [num_parts, 0, 0, 0, 0, 0, 0]
    nparts_this_file = [num_parts, 0, 0, 0, 0, 0, 0]
    can_have_types = [1, 0, 0, 0, 0, 0, 0]
    outfile["/Header"].attrs["NumPart_Total"] = nparts_total
    outfile["/Header"].attrs["NumPart_ThisFile"] = nparts_this_file
    outfile["/Header"].attrs["CanHaveTypes"] = can_have_types

    # re-write the cell metadata
    # currently only valid for n=0!!!
    outfile["/Cells/Counts/PartType0"][...] = 0
    if "Offsets" in outfile["/Cells"].keys():
        outfile["/Cells/Offsets/PartType0"][...] = 0
    if "OffsetsInFile" in outfile["/Cells"].keys():
        outfile["/Cells/OffsetsInFile/PartType0"][...] = 0

    # Tidy up
    infile.close()
    outfile.close()

    return
