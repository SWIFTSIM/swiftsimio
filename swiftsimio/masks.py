"""
Loading functions and objects that use masked information from the SWIFT
snapshots.
"""

import unyt
import h5py

import numpy as np

from swiftsimio import metadata, SWIFTMetadata, SWIFTUnits

from swiftsimio.accelerated import ranges_from_array


class SWIFTMask(object):
    """
    Main masking object. This can have masks for any present particle field in it.
    Pass in the SWIFTMetadata.

    Methods
    -------
    _generate_empty_masks(self)
        Create empty masks for all particles
    _unpack_cell_metadata(self)
        load cell metadata into local class variables
    constrain_mask( self, ptype: str, quantity: str, lower: unyt.array.unyt_quantity, upper: unyt.array.unyt_quantity,)
        constrains a particle mask based on the value of a the particle quantity
    _generate_cell_mask(self, restrict)
        generates spatially restricted mask for cell
    _update_spatial_mask(self, restrict, ptype: str, cell_mask: np.array)
        updates the particle mask using the cell mask. 
    constrain_spatial(self, restrict)
        generates spatially constrained cell mask
    convert_masks_to_ranges(self)
        converts the masks to range masks so that they take up less space.
    """

    def __init__(self, metadata: SWIFTMetadata, spatial_only=True):
        r"""
        SWIFTMask constructor

        Takes the SWIFT metadata and enables individual property-by-property masking
        when reading from snapshots. Please note that when masking like this
        order-in-file is not preserved, i.e. the 7th particle may not be the
        7th particle in the file.

        Parameters
        ----------
        metadata : SWIFTMetadata
            Metadata specifying masking for reading of snapshots

        spatial_only : bool, optional
            If True (the default), you can only constrain spatially. 
            However, this is significantly faster and considerably 
            more memory efficient (~ bytes per cell, rather than 
            ~ bytes per particle).

        """
        # ALEXEI: add examples, check if anything else needed for this doc

        self.metadata = metadata
        self.units = metadata.units
        self.spatial_only = spatial_only

        self._unpack_cell_metadata()

        if not spatial_only:
            self._generate_empty_masks()

    def _generate_empty_masks(self):
        r"""
        Generates the empty (i.e. all False) masks for all available particle
        types.
        """

        for ptype in self.metadata.present_particle_names:
            setattr(
                self, ptype, np.ones(getattr(self.metadata, f"n_{ptype}"), dtype=bool)
            )

        return

    def _unpack_cell_metadata(self):
        r"""
        Unpacks the cell metadata into local (to the class) variables. We do not
        read in information for empty cells.
        """

        # Reset this in case for any reason we have messed them up

        self.counts = {}
        self.offsets = {}

        with h5py.File(self.metadata.filename, "r") as handle:
            cell_handle = handle["Cells"]
            offset_handle = cell_handle["Offsets"]
            count_handle = cell_handle["Counts"]
            metadata_handle = cell_handle["Meta-data"]
            centers_handle = cell_handle["Centres"]

            # Only want to compute this once (even if it is fast, we do not
            # have a reliable stable sort in the case where cells do not
            # contain at least one of each type of particle).
            sort = None

            for ptype, pname in zip(
                self.metadata.present_particle_types,
                self.metadata.present_particle_names,
            ):
                part_type = f"PartType{ptype}"
                counts = count_handle[part_type][:]
                offsets = offset_handle[part_type][:]

                # When using MPI, we cannot assume that these are sorted.
                if sort is None:
                    # Only compute once; not stable between particle
                    # types if some datasets do not have particles in a cell!
                    sort = np.argsort(offsets)

                self.offsets[pname] = offsets[sort]
                self.counts[pname] = counts[sort]

            # Also need to sort centers in the same way
            self.centers = unyt.unyt_array(
                centers_handle[:][sort], units=self.units.length
            )

            # Note that we cannot assume that these are cubic, unfortunately.
            self.cell_size = unyt.unyt_array(
                metadata_handle.attrs["size"], units=self.units.length
            )

        return

    def constrain_mask(
        self,
        ptype: str,
        quantity: str,
        lower: unyt.array.unyt_quantity,
        upper: unyt.array.unyt_quantity,
    ):
        r"""
        Constrains the mask further for a given particle type, and bounds a 
        quantity between lower and upper values. 
        
        We update the mask such that

            lower < ptype.quantity <= upper

        The quantities must have units attached.

        Parameters
        ----------
        ptype : str
            particle type

        quantity : str
            quantity being constrained

        lower : unyt.array.unyt_quantity
            constraint lower bound 

        upper : unyt.array.unyt_quantity
            constraint upper bound

        See Also
        --------
        constrain_spatial : method to generate spatially constrained cell mask

        """

        if self.spatial_only:
            print("You cannot constrain a mask if spatial_only=True")
            print("Please re-initialise the SWIFTMask object with spatial_only=False")
            return

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
        r"""
        Generates spatially restricted mask for cell

        Takes the cell metadata and finds the mask for the _cells_ that are
        within the spatial region defined by the spatial mask. Not for
        user use. 

        Parameters
        ----------
        restrict : list
            Restrict is a 3 length list that contains length two arrays giving 
            the lower and upper bounds for that axis, e.g.

            restrict = [
                [0.5, 0.7],
                [0.1, 0.9],
                [0.0, 0.1]
            ]

            These values must have units associated with them.

        Returns
        -------
        cell_mask : np.array[bool]
            mask to indicate whether cells within specified spatial range
        """

        cell_mask = np.ones(len(self.centers), dtype=bool)

        for dimension in range(0, 3):
            if restrict[dimension] is None:
                continue

            # Include the cell size so it's easier to find the overlap
            lower = restrict[dimension][0] - 0.5 * self.cell_size[dimension]
            upper = restrict[dimension][1] + 0.5 * self.cell_size[dimension]
            boxsize = self.metadata.boxsize[dimension]

            # Now need to deal with the three wrapping cases:
            if lower.value < 0.0:
                # Wrap lower -> high
                lower += boxsize

                this_mask = np.logical_or.reduce(
                    [
                        self.centers[cell_mask, dimension] > lower,
                        self.centers[cell_mask, dimension] <= upper,
                    ]
                )
            elif upper > boxsize:
                # Wrap high -> lower
                upper -= boxsize

                this_mask = np.logical_or.reduce(
                    [
                        self.centers[cell_mask, dimension] > lower,
                        self.centers[cell_mask, dimension] <= upper,
                    ]
                )
            else:
                # No wrapping required
                this_mask = np.logical_and.reduce(
                    [
                        self.centers[cell_mask, dimension] > lower,
                        self.centers[cell_mask, dimension] <= upper,
                    ]
                )

            cell_mask[cell_mask] = this_mask

        return cell_mask

    def _update_spatial_mask(self, restrict, ptype: str, cell_mask: np.array):
        r"""
        Updates the particle mask using the cell mask. 
        
        We actually overwrite all non-used cells with False, rather than the 
        inverse, as we assume initially that we want to write all particles in, 
        and we want to respect other masks that may have been applied to the data.

        Parameters
        ----------
        restrict : list
            currently unused
        
        ptype : str
            particle type to update

        cell_mask : np.array
            cell mask used to update the particle mask
        """

        if self.spatial_only:
            counts = self.counts[ptype][cell_mask]
            offsets = self.offsets[ptype][cell_mask]

            this_mask = [[o, c + o] for c, o in zip(counts, offsets)]

            setattr(self, ptype, np.array(this_mask))
            setattr(self, f"{ptype}_size", np.sum(counts))

        else:
            counts = self.counts[ptype][~cell_mask]
            offsets = self.offsets[ptype][~cell_mask]

            # We must do the whole boolean mask business.
            this_mask = getattr(self, ptype)

            for count, offset in zip(counts, offsets):
                this_mask[offset : count + offset] = False

        return

    def constrain_spatial(self, restrict):
        r"""
        Uses the cell metadata to create a spatial mask. 
        
        This mask is necessarily approximate and is coarse-grained to the cell size.
        
        Parameters
        ----------
        restrict : list 
            length 3 list of length two arrays giving the lower and 
            upper bounds for that axis, e.g.

            restrict = [
                [0.5, 0.7],
                [0.1, 0.9],
                [0.0, 0.1]
                
            ]

            These values must have units associated with them. It is also acceptable
            to have a row as None to not restrict in this direction.

        See Also
        -------
        constrain_mask : method to further refine mask
        """

        cell_mask = self._generate_cell_mask(restrict)

        for ptype in self.metadata.present_particle_names:
            self._update_spatial_mask(restrict, ptype, cell_mask)

        return

    def convert_masks_to_ranges(self):
        r"""
        Converts the masks to range masks so that they take up less space.
        
        This is non-reversible. It is also not required, but can help save space
        on highly constrained machines before you start reading in the data.

        If you don't know what you are doing please don't use this.
        """

        if self.spatial_only:
            # We are already done!
            return
        else:
            # We must do the whole boolean mask stuff. To do that, we
            # First, convert each boolean mask into an integer mask
            # Use the accelerate.ranges_from_array function to convert
            # This into a set of ranges.

            for ptype in self.metadata.present_particle_names:
                setattr(
                    self,
                    ptype,
                    # Because it nests things in a list for some reason.
                    np.where(getattr(self, ptype))[0],
                )

                setattr(self, f"{ptype}_size", getattr(self, ptype).size)

            for ptype in self.metadata.present_particle_names:
                setattr(self, ptype, ranges_from_array(getattr(self, ptype)))

        return
