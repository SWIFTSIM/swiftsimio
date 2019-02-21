"""
Loading functions and objects that use masked information from the SWIFT
snapshots.
"""

import unyt
import h5py

import numpy as np

from swiftsimio import metadata, SWIFTMetadata, SWIFTUnits


class SWIFTMask(object):
    """
    Main masking object. This can have masks for any present particle field in it.
    Pass in the SWIFTMetadata.
    """

    def __init__(self, metadata: SWIFTMetadata):
        """
        Takes the SWIFT metadata and enables individual property-by-property masking
        when reading from snapshots. Please note that when masking like this
        order-in-file is not preserved, i.e. the 7th particle may not be the
        7th particle in the file.
        """

        self.metadata = metadata
        self.units = metadata.units

        self._unpack_cell_metadata()

        self._generate_empty_masks()

    def _generate_empty_masks(self):
        """
        Generates the empty (i.e. all True) masks for all available particle
        types.
        """

        for ptype in self.metadata.present_particle_names:
            setattr(
                self, ptype, np.ones(getattr(self.metadata, f"n_{ptype}"), dtype=bool)
            )

        return

    def _unpack_cell_metadata(self):
        """
        Unpacks the cell metadata into local (to the class) variables. We do not
        read in information for empty cells.
        """

        # Reset this in case for any reason we have messed them up

        self.counts = {}
        self.offsets = {}

        # First find the non-empty cells.

        with h5py.File(self.metadata.filename, "r") as handle:
            non_empty_mask = np.logical_and.reduce(
                [
                    handle["Cells"]["Counts"][f"PartType{ptype}"][...] != 0
                    for ptype in self.metadata.present_particle_types
                ]
            )

            for ptype, pname in zip(
                self.metadata.present_particle_types,
                self.metadata.present_particle_names,
            ):
                self.counts[pname] = handle["Cells"]["Counts"][f"PartType{ptype}"][
                    non_empty_mask
                ]
                self.offsets[pname] = handle["Cells"]["Offsets"][f"PartType{ptype}"][
                    non_empty_mask
                ]

            self.centers = (
                handle["Cells"]["Centres"][non_empty_mask, :] * self.units.length
            )

            # Note that we cannot assume that these are cubic, unfortunately.
            self.cell_size = (
                np.array(handle["Cells"]["Meta-data"].attrs["size"]) * self.units.length
            )

        return

    def constrain_mask(
        self,
        ptype: str,
        quantity: str,
        lower: unyt.array.unyt_quantity,
        upper: unyt.array.unyt_quantity,
    ):
        """
        Constrains the mask further for a given particle type, and bounds a 
        quantity between lower and upper values. We update the mask such
        that

            lower < ptype.quantity <= upper

        Note that the quantities must have units attached.
        """

        current_mask = getattr(self, ptype)

        handle = {v: k for k, v in getattr(metadata.particle_fields, ptype).items()}[
            quantity
        ]
        unit = getattr(self.units, ptype)[quantity]
        # We use the type and not the number because it is far easier for users to understand.
        particle_number = {
            v: k for k, v in metadata.particle_types.particle_name_underscores.items()
        }[ptype]
        # Load in the relevant data.

        with h5py.File(self.metadata.filename, "r") as file:
            # Surprisingly this is faster than just using the boolean
            # indexing because h5py has slow indexing routines.
            data = (
                np.take(
                    file[f"PartType{particle_number}/{handle}"],
                    np.where(current_mask)[0],
                    axis=0,
                )
                * unit
            )

        new_mask = np.logical_and.reduce([data > lower, data <= upper])

        current_mask[current_mask] = new_mask

        setattr(self, ptype, current_mask)

        return

    def _generate_cell_mask(self, restrict):
        """
        Takes the cell metadata and finds the mask for the _cells_ that are
        within the spatial region defined by the spatial mask. Not for
        user use. 

        Uses the cell metadata to create a spatial mask. Restrict is a 3 length
        list that contains length two arrays giving the lower and upper bounds
        for that axis, e.g.

        restrict = [
            [0.5, 0.7],
            [0.1, 0.9],
            [0.0, 0.1]
        ]

        These values must have units associated with them.
        """

        cell_mask = np.ones(len(self.centers), dtype=bool)

        for dimension in range(0, 3):
            if restrict[dimension] is None:
                continue

            lower, upper = restrict[dimension]

            # Now include the cell size.
            lower -= 0.5 * self.cell_size[dimension]
            upper += 0.5 * self.cell_size[dimension]

            this_mask = np.logical_and.reduce(
                [
                    self.centers[cell_mask, dimension] > lower,
                    self.centers[cell_mask, dimension] <= upper,
                ]
            )

            cell_mask[cell_mask] = this_mask

        return cell_mask

    def _update_spatial_mask(self, restrict, ptype: str, cell_mask: np.array):
        """
        Generates the mask for the _particles_ of all types. Not for user
        use.

        Uses the cell metadata to create a spatial mask. Restrict is a 3 length
        list that contains length two arrays giving the lower and upper bounds
        for that axis, e.g.

        restrict = [
            [0.5, 0.7],
            [0.1, 0.9],
            [0.0, 0.1]
        ]

        These values must have units associated with them.
        """

        counts = self.counts[ptype][cell_mask]
        offsets = self.offsets[ptype][cell_mask]

        mask = np.zeros_like(getattr(self, ptype))

        for count, offset in zip(counts, offsets):
            mask[offset : count + offset] = True

        setattr(self, ptype, np.logical_and.reduce([mask, getattr(self, ptype)]))

        return

    def constrain_spatial(self, restrict):
        """
        Uses the cell metadata to create a spatial mask. Restrict is a 3 length
        list that contains length two arrays giving the lower and upper bounds
        for that axis, e.g.

        restrict = [
            [0.5, 0.7],
            [0.1, 0.9],
            [0.0, 0.1]
        ]

        These values must have units associated with them. It is also acceptable
        to have a row as None to not restrict in this direction.

        Please note that this is approximate and is coarse-grained to the cell size.
        
        If you would like to further refine this afterwards, please use the
        constrain_mask method.
        """

        cell_mask = self._generate_cell_mask(restrict)

        for ptype in self.metadata.present_particle_names:
            self._update_spatial_mask(restrict, ptype, cell_mask)

        return

    def convert_masks_to_integer(self):
        """
        Converts the masks to integer masks so that they take up less space.
        
        This is non-reversible. It is also not required, but can help save space
        on highly constrained machines before you start reading in the data.

        If you don't know what you are doing please don't use this.
        """

        for ptype in self.metadata.present_particle_names:
            setattr(
                self,
                ptype,
                # Because it nests things in a list for some reason.
                np.where(getattr(self, ptype))[0],
            )

        return
