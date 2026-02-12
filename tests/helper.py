"""Contains helper functions for the test routines."""

import pytest
import numpy as np
import h5py
import unyt as u
from swiftsimio.subset_writer import find_links, write_metadata
from swiftsimio import mask, cosmo_array, Writer
from swiftsimio.units import cosmo_units
from swiftsimio.masks import SWIFTMask


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


def create_minimal_writer(a: float = 1.0, n_p: int = 32**3, lbox: float = 100):
    """
    Create a :class:`~swiftsimio.snapshot_writer.Writer` with required gas fields.

    Parameters
    ----------
    a : float, optional
        The scale factor.

    n_p : int, optional
        The number of particles.

    lbox : float, optional
        The box side length in Mpc.

    Returns
    -------
    ~swiftsimio.snapshot_writer.Writer
        The writer with required gas fields initialized.
    """
    # Box is 100 Mpc
    boxsize = cosmo_array(
        [lbox, lbox, lbox],
        u.Mpc,
        comoving=True,
        scale_factor=a,
        scale_exponent=1,
    )

    # Generate object. cosmo_units corresponds to default Gadget-oid units
    # of 10^10 Msun, Mpc, and km/s
    w = Writer(unit_system=cosmo_units, boxsize=boxsize, scale_factor=a)

    # Randomly spaced coordinates from 0, lbox Mpc in each direction
    w.gas.coordinates = cosmo_array(
        np.random.rand(n_p, 3) * lbox,
        u.Mpc,
        comoving=True,
        scale_factor=w.scale_factor,
        scale_exponent=1,
    )

    # Random velocities from 0 to 1 km/s
    w.gas.velocities = cosmo_array(
        np.random.rand(n_p, 3),
        u.km / u.s,
        comoving=True,
        scale_factor=w.scale_factor,
        scale_exponent=1,
    )

    # Generate uniform masses as 10^6 solar masses for each particle
    w.gas.masses = cosmo_array(
        np.ones(n_p, dtype=float) * 1e6,
        u.msun,
        comoving=True,
        scale_factor=w.scale_factor,
        scale_exponent=0,
    )

    # Generate internal energy corresponding to 10^4 K
    w.gas.internal_energy = cosmo_array(
        np.ones(n_p, dtype=float) * 1e4 / 1e6,
        u.kb * u.K / u.solMass,
        comoving=True,
        scale_factor=w.scale_factor,
        scale_exponent=-2,
    )

    # Generate initial guess for smoothing lengths based on MIPS
    w.gas.generate_smoothing_lengths()

    # IDs will be auto-generated on file write
    return w


def create_two_type_writer(a: float = 1.0, n_p: int = 32**3, lbox: float = 100):
    """
    Create a :class:`~swiftsimio.snapshot_writer.Writer` with required gas & DM fields.

    Parameters
    ----------
    a : float, optional
        The scale factor.

    n_p : int, optional
        The number of particles.

    lbox : float, optional
        The box side length in Mpc.

    Returns
    -------
    ~swiftsimio.snapshot_writer.Writer
        The writer with required gas and dark_matter fields initialized.
    """
    w = create_minimal_writer(a=a, n_p=n_p, lbox=lbox)

    # Randomly spaced coordinates from 0, lbox Mpc in each direction
    w.dark_matter.coordinates = cosmo_array(
        np.random.rand(n_p, 3) * lbox,
        u.Mpc,
        comoving=True,
        scale_factor=w.scale_factor,
        scale_exponent=1,
    )

    # Random velocities from 0 to 1 km/s
    w.dark_matter.velocities = cosmo_array(
        np.random.rand(n_p, 3),
        u.km / u.s,
        comoving=True,
        scale_factor=w.scale_factor,
        scale_exponent=1,
    )

    # Generate uniform masses as 10^6 solar masses for each particle
    w.dark_matter.masses = cosmo_array(
        np.ones(n_p, dtype=float) * 1e6,
        u.msun,
        comoving=True,
        scale_factor=w.scale_factor,
        scale_exponent=0,
    )

    # IDs will be auto-generated on file write
    return w
