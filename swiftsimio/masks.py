"""
Loading functions and objects that use masked information from the SWIFT
snapshots.
"""

import warnings

import unyt
import h5py

import numpy as np

from swiftsimio import SWIFTMetadata

from swiftsimio.objects import InvalidSnapshot

from swiftsimio.accelerated import ranges_from_array
from typing import Dict


class SWIFTMask(object):
    """
    Main masking object. This can have masks for any present particle field in it.
    Pass in the SWIFTMetadata.
    """

    group_mapping: dict | None = None
    group_size_mapping: dict | None = None

    def __init__(self, metadata: SWIFTMetadata, spatial_only=True):
        """
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

        self.metadata = metadata
        self.units = metadata.units
        self.spatial_only = spatial_only

        if not self.metadata.masking_valid:
            raise NotImplementedError(
                f"Masking not supported for {self.metadata.output_type} filetype"
            )

        if self.metadata.partial_snapshot:
            raise InvalidSnapshot(
                "You cannot use masks on partial snapshots. Please use the virtual "
                "file generated by SWIFT (use snapshot.hdf5, not snapshot.0.hdf5)."
            )

        self._unpack_cell_metadata()

        if not spatial_only:
            self._generate_empty_masks()

    def _generate_mapping_dictionary(self) -> dict[str, str]:
        """
        Creates cross-links between 'group names' and their underlying cell metadata
        names. Allows for pointers to be used instead of re-creating masks.
        """

        if self.group_mapping is not None:
            return self.group_mapping

        if self.metadata.shared_cell_counts is None:
            # Each and every particle type has its own cell counts, offsets,
            # and hence masks.
            self.group_mapping = {
                group: f"_{group}" for group in self.metadata.present_group_names
            }
        else:
            # We actually only have _one_ mask!
            self.group_mapping = {
                group: "_shared" for group in self.metadata.present_group_names
            }

        return self.group_mapping

    def _generate_size_mapping_dictionary(self) -> dict[str, str]:
        """
        Creates cross-links between 'group names' and their underlying cell metadata
        names. Allows for pointers to be used instead of re-creating masks.
        """

        if self.group_size_mapping is not None:
            return self.group_size_mapping

        if self.metadata.shared_cell_counts is None:
            # Each and every particle type has its own cell counts, offsets,
            # and hence masks.
            self.group_size_mapping = {
                f"{group}_size": f"_{group}_size"
                for group in self.metadata.present_group_names
            }
        else:
            # We actually only have _one_ mask!
            self.group_size_mapping = {
                f"{group}_size": "_shared_size"
                for group in self.metadata.present_group_names
            }

        return self.group_size_mapping

    def _generate_update_list(self) -> list[str]:
        """
        Gets a list of internal mask variables that need to be updated when
        we change the spatial mask.
        """

        if self.metadata.shared_cell_counts is None:
            # Each and every particle type has its own cell counts, offsets,
            # and hence masks.
            return [f"_{group}" for group in self.metadata.present_group_names]
        else:
            # We actually only have _one_ mask!
            return ["_shared"]

    def __getattr__(self, name):
        """
        Overloads the getattr method to allow for direct access to the masks
        for each particle type.
        """
        mappings = {
            **self._generate_mapping_dictionary(),
            **self._generate_size_mapping_dictionary(),
        }

        underlying_name = mappings.get(name, None)

        if underlying_name is not None:
            return getattr(self, underlying_name)

        raise AttributeError(f"Attribute {name} not found in SWIFTMask")

    def _generate_empty_masks(self):
        """
        Generates the empty (i.e. all False) masks for all available particle
        types.
        """

        mapping = self._generate_mapping_dictionary()

        if self.metadata.shared_cell_counts is not None:
            size = getattr(
                self.metadata, f"n_{self.metadata.shared_cell_counts.lower()}"
            )
            self._shared = np.ones(size, dtype=bool)
            self._shared_size = size

        else:
            # Create empty masks for each and every particle type.
            for group_name, data_name in mapping.items():
                size = getattr(self.metadata, f"n_{group_name}")
                setattr(self, data_name, np.ones(size, dtype=bool))
                setattr(self, f"{data_name}_size", size)

        return

    def _unpack_cell_metadata(self):
        """
        Unpacks the cell metadata into local (to the class) variables. We do not
        read in information for empty cells.
        """

        # Reset this in case for any reason we have messed them up

        self.counts = {}
        self.offsets = {}

        cell_handle = self.units.handle["Cells"]
        count_handle = cell_handle["Counts"]
        metadata_handle = cell_handle["Meta-data"]
        centers_handle = cell_handle["Centres"]

        try:
            offset_handle = cell_handle["OffsetsInFile"]
        except KeyError:
            # Previous version of SWIFT did not have distributed
            # file i/o implemented
            offset_handle = cell_handle["Offsets"]

        if self.metadata.shared_cell_counts is not None:
            # Single - called _shared.
            self.offsets["shared"] = offset_handle[self.metadata.shared_cell_counts][:]
            self.counts["shared"] = count_handle[self.metadata.shared_cell_counts][:]
        else:
            for group, group_name in zip(
                self.metadata.present_groups, self.metadata.present_group_names
            ):
                counts = count_handle[group][:]
                offsets = offset_handle[group][:]

                self.offsets[group_name] = offset_handle[group][:]
                self.counts[group_name] = count_handle[group][:]

        # Only want to compute this once (even if it is fast, we do not
        # have a reliable stable sort in the case where cells do not
        # contain at least one of each type of particle).
        sort = None

        # Now perform sort:
        for key in self.offsets.keys():
            offsets = self.offsets[key]
            counts = self.counts[key]

            # When using MPI, we cannot assume that these are sorted.
            if sort is None:
                # Only compute once; not stable between particle
                # types if some datasets do not have particles in a cell!
                sort = np.argsort(offsets)

            self.offsets[key] = offsets[sort]
            self.counts[key] = counts[sort]

        # Also need to sort centers in the same way
        self.centers = unyt.unyt_array(centers_handle[:][sort], units=self.units.length)

        # Note that we cannot assume that these are cubic, unfortunately.
        self.cell_size = unyt.unyt_array(
            metadata_handle.attrs["size"], units=self.units.length
        )

        return

    def constrain_mask(
        self,
        group_name: str,
        quantity: str,
        lower: unyt.array.unyt_quantity,
        upper: unyt.array.unyt_quantity,
    ):
        """
        Constrains the mask further for a given particle type, and bounds a
        quantity between lower and upper values.

        We update the mask such that

            lower < group_name.quantity <= upper

        The quantities must have units attached.

        Parameters
        ----------
        group_name : str
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
            raise ValueError(
                "You cannot constrain a mask if spatial_only=True. "
                "Please re-initialise the SWIFTMask object with spatial_only=False"
            )

        mapping = self._generate_mapping_dictionary()
        data_name = mapping[group_name]

        current_mask = getattr(self, data_name)

        group_metadata = getattr(self.metadata, f"{group_name}_properties")
        unit_dict = {
            k: v for k, v in zip(group_metadata.field_names, group_metadata.field_units)
        }

        unit = unit_dict[quantity]

        handle_dict = {
            k: v for k, v in zip(group_metadata.field_names, group_metadata.field_paths)
        }

        handle = handle_dict[quantity]

        # Load in the relevant data.

        with h5py.File(self.metadata.filename, "r") as file:
            # Surprisingly this is faster than just using the boolean
            # indexing because h5py has slow indexing routines.
            data = unyt.unyt_array(
                np.take(file[handle], np.where(current_mask)[0], axis=0), units=unit
            )

        new_mask = np.logical_and.reduce([data > lower, data <= upper])

        current_mask[current_mask] = new_mask

        setattr(self, data_name, current_mask)

        return

    def _generate_cell_mask(self, restrict):
        """
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
            mask to indicate which cells are within the specified spatial range
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

    def _update_spatial_mask(self, restrict, data_name: str, cell_mask: np.array):
        """
        Updates the particle mask using the cell mask.

        We actually overwrite all non-used cells with False, rather than the
        inverse, as we assume initially that we want to write all particles in,
        and we want to respect other masks that may have been applied to the data.

        Parameters
        ----------

        restrict : list
            currently unused

        data_name : str
            underlying data to update (e.g. _gas, _shared)

        cell_mask : np.array
            cell mask used to update the particle mask
        """

        count_name = data_name[1:]  # Remove the underscore

        if self.spatial_only:
            counts = self.counts[count_name][cell_mask]
            offsets = self.offsets[count_name][cell_mask]

            this_mask = [[o, c + o] for c, o in zip(counts, offsets)]

            setattr(self, data_name, np.array(this_mask))
            setattr(self, f"{data_name}_size", np.sum(counts))

        else:
            counts = self.counts[count_name][~cell_mask]
            offsets = self.offsets[count_name][~cell_mask]

            # We must do the whole boolean mask business.
            this_mask = getattr(self, data_name)

            for count, offset in zip(counts, offsets):
                this_mask[offset : count + offset] = False

        return

    def constrain_spatial(self, restrict, intersect: bool = False):
        """
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

        intersect : bool
            If `True`, intersect the spatial mask with any existing spatial mask to
            select two (or more) regions with repeated calls to `constrain_spatial`.
            By default (`False`) any existing mask is overwritten.

        See Also
        -------

        constrain_mask : method to further refine mask
        """

        if hasattr(self, "cell_mask") and intersect:
            # we already have a mask and are in intersect mode
            self.cell_mask = np.logical_or(
                self.cell_mask, self._generate_cell_mask(restrict)
            )
        else:
            # we just make a new mask
            self.cell_mask = self._generate_cell_mask(restrict)

        for mask in self._generate_update_list():
            self._update_spatial_mask(restrict, mask, self.cell_mask)

        return

    def convert_masks_to_ranges(self):
        """
        Converts the masks to range masks so that they take up less space.

        This is non-reversible. It is also not required, but can help save space
        on highly constrained machines before you start reading in the data.

        If you don't know what you are doing please don't use this.
        """

        # Spatial only already comes like this!
        if not self.spatial_only:
            # We must do the whole boolean mask stuff. To do that, we
            # First, convert each boolean mask into an integer mask
            # Use the accelerate.ranges_from_array function to convert
            # This into a set of ranges.
            for mask in self._generate_update_list():
                where_array = np.where(getattr(self, mask))[0]
                setattr(self, f"{mask}_size", where_array.size)
                setattr(self, mask, ranges_from_array(where_array))

        return

    def constrain_index(self, index: int):
        """
        Constrain the mask to a single row.

        Intended for use with SOAP catalogues, mask to read only a single row.

        Parameters
        ----------
        index : int
            The index of the row to select.
        """
        if not self.metadata.filetype == "SOAP":
            warnings.warn("Not masking a SOAP catalogue, nothing constrained.")
            return
        for group_name in self.metadata.present_group_names:
            setattr(self, group_name, np.array([[index, index + 1]]))
            setattr(self, f"{group_name}_size", 1)
        return

    def get_masked_counts_offsets(
        self,
    ) -> tuple[dict[str, np.array], dict[str, np.array]]:
        """
        Returns the particle counts and offsets in cells selected by the mask

        Returns
        -------

        Dict[str, np.array], Dict[str, np.array]
            Dictionaries containing the particle offets and counts for each particle
            type. For example, the particle counts dictionary would be of the form

            .. code-block:: python

                {"gas": [g_0, g_1, ...],
                 "dark matter": [bh_0, bh_1, ...], ...}

            where the keys would be each of the particle types and values are arrays
            of the number of corresponding particles in each cell (in this case there
            would be g_0 gas particles in the first cell, g_1 in the second, etc.).
            The structure of the dictionaries is the same for the offsets, with the
            arrays now storing the offset of the first particle in the cell.

        """
        if self.spatial_only:
            masked_counts = {}
            current_offsets = {}
            if not hasattr(self, "cell_mask"):
                raise RuntimeError(
                    "Subset writing requires specifying a cell mask. Please use "
                    "constrain_spatial with a suitable restrict array to generate one."
                )
            for part_type, counts in self.counts.items():
                masked_counts[part_type] = counts * self.cell_mask

                current_offsets[part_type] = [0] * counts.size
                running_sum = 0
                for i in range(len(counts)):
                    current_offsets[part_type][i] = running_sum
                    running_sum += masked_counts[part_type][i]

            return masked_counts, current_offsets
        else:
            raise ("Only applies on spatial only masks")
