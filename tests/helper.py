"""Contains helper functions for the test routines."""

import pytest
import numpy as np
import h5py
import unyt as u
from swiftsimio.subset_writer import find_links, write_metadata
from swiftsimio import mask, cosmo_array
from swiftsimio.masks import SWIFTMask
import swiftsimio.metadata.unit.unit_fields as swuf


def _mask_without_warning(fname: str, **kwargs: dict) -> SWIFTMask:
    """
    Create a mask suppressing expected warnings.

    Parameters
    ----------
    fname : str
        File name to mask.

    **kwargs : dict
        Arbitrary additional kwargs.

    Returns
    -------
    SWIFTMask
        The mask object.
    """
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


def create_in_memory_hdf5(filename: str = "f1") -> h5py.File:
    """
    Create an in-memory hdf5 file object.

    Parameters
    ----------
    filename : str
        Name for the memory-backed file.

    Returns
    -------
    h5py.File
        A memory-backed HDF5 dataset file.
    """
    return h5py.File(filename, driver="core", mode="a", backing_store=False)


def create_n_particle_dataset(
    filename: str, output_name: str, num_parts: int = 2
) -> None:
    """
    Create an hdf5 snapshot with a desired number of identical particles.

    Parameters
    ----------
    filename : str
        Name of file from which to copy metadata.

    output_name : str
        Name of single particle snapshot.

    num_parts : int
        Number of particles to create (default: 2).
    """
    # Create a mask:
    # - in order to write metadata
    # - to find the relevant cell in the cell metadata
    data_mask = _mask_without_warning(filename, safe_padding=False)
    region = cosmo_array(
        [[0.999, 1.001]] * 3,
        data_mask.metadata.units.length,
        comoving=True,
        scale_factor=data_mask.metadata.a,
        scale_exponent=1,
    )
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
    mean_h = np.mean(infile["/PartType0/SmoothingLengths"])
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
    del outfile["/Cells/Counts/PartType1"]
    if "Offsets" in outfile["/Cells"].keys():
        del outfile["/Cells/Offsets/PartType1"]
    if "OffsetsInFile" in outfile["/Cells"].keys():
        del outfile["/Cells/OffsetsInFile/PartType1"]
    nparts_total = [num_parts, 0, 0, 0, 0, 0, 0]
    nparts_this_file = [num_parts, 0, 0, 0, 0, 0, 0]
    can_have_types = [1, 0, 0, 0, 0, 0, 0]
    outfile["/Header"].attrs["NumPart_Total"] = nparts_total
    outfile["/Header"].attrs["NumPart_ThisFile"] = nparts_this_file
    outfile["/Header"].attrs["CanHaveTypes"] = can_have_types

    # re-write the cell metadata
    outfile["/Cells/Counts/PartType0"][...] = np.where(
        data_mask.cell_mask["gas"], num_parts, 0
    )
    if "Offsets" in outfile["/Cells"].keys():
        outfile["/Cells/Offsets/PartType0"][...] = 0
        outfile["/Cells/Offsets/PartType0"][
            np.argwhere(data_mask.cell_mask["gas"]).squeeze() + 1 :
        ] = num_parts
    if "OffsetsInFile" in outfile["/Cells"].keys():
        outfile["/Cells/OffsetsInFile/PartType0"][...] = 0
        outfile["/Cells/OffsetsInFile/PartType0"][
            np.argwhere(data_mask.cell_mask["gas"]).squeeze() + 1 :
        ] = num_parts

    # Tidy up
    infile.close()
    outfile.close()

    return


def extra_type_generate_units(
    mass: u.Unit, length: u.Unit, time: u.Unit, current: u.Unit, temperature: u.Unit
) -> dict[str, dict[str, u.Unit]]:
    """
    Generate units differently for testing.

    This function is used to override the inbuilt swiftsimio generate_units function from
    metadata.unit.unit_fields. This allows the specification of a new particle type and
    the values and types associated with that type.

    Parameters
    ----------
    mass : Unit
        The mass unit.

    length : Unit
        The length unit.

    time : Unit
        The time unit.

    current : Unit
        The current unit.

    temperature : Unit
        The temperature unit.

    Returns
    -------
    dict[str, Unit]
        A dictionary mapping field names to units.
    """
    dict_out = swuf.generate_units(mass, length, time, current, temperature)

    extratype = {
        "coordinates": length,
        "masses": mass,
        "particle_ids": None,
        "velocities": length / time,
        "potential": length * length / (time * time),
        "density": mass / (length**3),
        "entropy": mass * length**2 / (time**2 * temperature),
        "internal_energy": (length / time) ** 2,
        "smoothing_length": length,
        "pressure": mass / (length * time**2),
        "diffusion": None,
        "sfr": mass / time,
        "temperature": temperature,
        "viscosity": None,
        "specific_sfr": 1 / time,
        "material_id": None,
        "radiated_energy": mass * (length / time) ** 2,
    }

    dict_out["extratype"] = extratype
    return dict_out
