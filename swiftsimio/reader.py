"""
This file contains four major objects:

+ SWIFTUnits, which is a unit system that can be queried for units (and converts arrays
  to relevant unyt arrays when read from the HDF5 file)
+ SWIFTMetadata, which contains all of the metadata from the file
+ __SWIFTGroupDataset, which contains particle information but should never be
  directly accessed. Use generate_dataset to create one of these. The reasoning
  here is that properties can only be added to the class afterwards, and not
  directly in an _instance_ of the class.
+ SWIFTDataset, a container class for all of the above.
"""


from swiftsimio.accelerated import read_ranges_from_file
from swiftsimio.objects import cosmo_array, cosmo_factor, a

from swiftsimio.metadata.objects import (
    metadata_discriminator,
    SWIFTUnits,
    SWIFTGroupMetadata,
    SWIFTMetadata,
)

import re
import h5py
import unyt
import numpy as np
import warnings

from datetime import datetime
from pathlib import Path

from typing import Union, Callable, List, Optional


def generate_getter(
    filename,
    name: str,
    field: str,
    unit: unyt.unyt_quantity,
    mask: Union[None, np.ndarray],
    mask_size: int,
    cosmo_factor: cosmo_factor,
    description: str,
    compression: str,
    physical: bool,
    valid_transform: bool,
    columns: Union[None, slice] = None,
):
    """
    Generates a function that:

    a) If self._`name` exists, return it
    b) If not, open `filename`
    c) Reads filename[`field`]
    d) Set self._`name`
    e) Return self._`name`.

    Parameters
    ----------

    filename: str
        Filename of the HDF5 file that everything will be read from. Used to generate
        the HDF5 dataset.

    name: str
        Output name (snake_case) of the field.

    field: str
        Full path of field, including e.g. particle type. Examples include
        ``/PartType0/Velocities``.

    unit: unyt.unyt_quantity
        Output unit of the resultant ``cosmo_array``

    mask: None or np.ndarray
        Mask to be used with ``accelerated.read_ranges_from_file``, i.e. an array of
        integers that describe ranges to be read from the file.

    mask_size: int
        Size of the mask if present.

    cosmo_factor: cosmo_factor
        Cosmology factor object corresponding to this array.

    description: str
        Description (read from HDF5 file) of the data.

    compression: str
        String describing the lossy compression filters that were applied to the
        data (read from the HDF5 file).

    physical: bool
        Bool that describes whether the data in the file is stored in comoving
        or physical units.

    valid_transform: bool
        Bool that describes whether converting this field from physical to comoving
        units is a valid operation.

    columns: np.lib.index_tricks.IndexEpression, optional
        Index expression corresponding to which columns to read from the numpy array.
        If not provided, we read all columns and return an n-dimensional array.


    Returns
    -------

    getter: callable
        A callable object that gets the value of the array that has been saved to
        ``_name``. This function takes only ``self`` from the
        :obj:``__SWIFTGroupDataset`` class.


    Notes
    -----

    The major use of this function is for its side effect of setting ``_name`` as
    a member of the class on first read. When the attribute is accessed, it will
    be dynamically read from the file, to keep initial memory usage as minimal
    as possible.

    If the resultant array is modified, it will not be re-read from the file.

    """

    # Must do this _outside_ getter because of weird locality issues with the
    # use of None as the default.
    # Here, we need to ensure that in the cases where we're using columns,
    # during a partial read, that we respect the single-column dataset nature.
    use_columns = columns is not None

    if not use_columns:
        columns = np.s_[:]

    def getter(self):
        current_value = getattr(self, f"_{name}")

        if current_value is not None:
            return current_value
        else:
            with h5py.File(filename, "r") as handle:
                try:
                    if mask is not None:
                        # First, need to calculate data shape (which may be
                        # non-trivial), so we read in the first value
                        first_value = handle[field][0]

                        output_type = first_value.dtype
                        output_size = first_value.size

                        if output_size != 1 and not use_columns:
                            output_shape = (mask_size, output_size)
                        else:
                            output_shape = mask_size

                        setattr(
                            self,
                            f"_{name}",
                            cosmo_array(
                                read_ranges_from_file(
                                    handle[field],
                                    mask,
                                    output_shape=output_shape,
                                    output_type=output_type,
                                    columns=columns,
                                ),
                                unit,
                                cosmo_factor=cosmo_factor,
                                name=description,
                                compression=compression,
                                comoving=not physical,
                                valid_transform=valid_transform,
                            ),
                        )
                    else:
                        setattr(
                            self,
                            f"_{name}",
                            cosmo_array(
                                # Only use column data if array is multidimensional, otherwise
                                # we will crash here
                                handle[field][:, columns]
                                if handle[field].ndim > 1
                                else handle[field][:],
                                unit,
                                cosmo_factor=cosmo_factor,
                                name=description,
                                compression=compression,
                                comoving=not physical,
                                valid_transform=valid_transform,
                            ),
                        )
                except KeyError:
                    print(f"Could not read {field}")
                    return None

        return getattr(self, f"_{name}")

    return getter


def generate_setter(name: str):
    """
    Generates a function that sets self._name to the value that is passed to it.

    Parameters
    ----------
    name : str
        the name of the attribute to set

    Returns
    -------
    setter : callable
        a callable object that sets the attribute specified by ``name`` to the value
        passed to it.
    """

    def setter(self, value):
        setattr(self, f"_{name}", value)

        return

    return setter


def generate_deleter(name: str):
    """
    Generates a function that destroys self._name (sets it back to None).

    Parameters
    ----------
    name : str
        the name of the field to be destroyed

    Returns
    -------
    deleter : callable
        callable that destroys ``name`` field
    """

    def deleter(self):
        current_value = getattr(self, f"_{name}")
        del current_value
        setattr(self, f"_{name}", None)

        return

    return deleter


class __SWIFTGroupDataset(object):
    """
    Creates empty property fields

    Do not use this class alone; it is essentially completely empty. It is filled
    with properties by generate_dataset.

    Methods
    -------
    generate_empty_properties(self)
        creates empty properties to be accessed through setter and getter functions
    """

    def __init__(self, group_metadata: SWIFTGroupMetadata):
        """
        Constructor for SWIFTGroupDatasets class

        This function primarily calls the generate_empty_properties
        function to ensure that defaults are set correctly.

        Parameters
        ----------
        group_metadata : SWIFTGroupMetadata
            the metadata used to generate empty properties
        """
        self.filename = group_metadata.filename
        self.units = group_metadata.units

        self.group = group_metadata.group
        self.group_name = group_metadata.group_name

        self.group_metadata = group_metadata
        self.metadata = group_metadata.metadata

        self.generate_empty_properties()

        return

    def generate_empty_properties(self):
        """
        Generates the empty properties that will be accessed through the
        setter and getters.

        Initially set all of the _{name} values to None. If it doesn't
        _exist_ in the file, the variable is not created.
        """

        for field_name, field_path in zip(
            self.group_metadata.field_names, self.group_metadata.field_paths
        ):
            if field_path in self.metadata.handle:
                setattr(self, f"_{field_name}", None)
            else:
                raise AttributeError(
                    (
                        f"Cannot find attribute {field_path} in file although"
                        "it was present when searching initially."
                    )
                )

        return


class __SWIFTNamedColumnDataset(object):
    """
    Holder class for individual named datasets. Very similar to
    __SWIFTGroupsDatasets but much simpler.
    """

    def __init__(self, field_path: str, named_columns: List[str], name: str):
        r"""
        Constructor for __SWIFTNamedColumnDataset class

        Parameters
        ----------
        field_path : str
            path to field within hdf5 snapshot
        named_columns : list of str
            list of categories for the variable `name`
        name : str
            the variable of interest

        Examples
        --------
        For a gas particle we might be interested in the mass fractions for a number
        of elements (e.g. hydrogen, helium, carbon, etc). In a SWIFT snapshot these
        would be found in `field_path` = /PartType0/ElementMassFractions. The
        `named_columns` would be the list of elements (["hydrogen", ...]) and the
        variable we're interested in is the mass fraction `name` = element_mass_fraction.
        Thus,
            data.gas = __SWIFTNamedColumnDataset(
            "/PartType0/ElementMassFractions",
            ["hydrogen", "helium"],
            "element_mass_fraction"
            )
        would make visible:
            data.gas.element_mass_fraction.hydrogen
            data.gas.element_mass_fraction.helium
        """
        self.field_path = field_path
        self.named_columns = named_columns
        self.name = name

        # Need to initialise for the getter() call.
        for column in named_columns:
            setattr(self, f"_{column}", None)

        return

    def __str__(self):
        return f'Named columns instance with {self.named_columns} available for "{self.name}"'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.named_columns)

    def __eq__(self, other):
        return self.named_columns == other.named_columns and self.name == other.name


def generate_datasets(group_metadata: SWIFTGroupMetadata, mask):
    """
    Generates a SWIFTGroupDatasets _class_ that corresponds to the
    particle type given.

    We _must_ do the following _outside_ of the class itself, as one
    can assign properties to a _class_ but not _within_ a class
    dynamically.

    Here we loop through all of the possible properties in the metadata file.
    We then use the builtin property() function and some generators to
    create setters and getters for those properties. This will allow them
    to be accessed from outside by using SWIFTGroupDatasets.name, where
    the name is, for example, coordinates.

    Parameters
    ----------
    group_metadata : SWIFTGroupMetadata
        the metadata for the group
    mask : SWIFTMask
        the mask object for the datasets
    """

    filename = group_metadata.filename
    group = group_metadata.group
    group_name = group_metadata.group_name

    group_nice_name = group_metadata.metadata.get_nice_name(group)

    # Mask is an object that contains all masks for all possible datasets.
    if mask is not None:
        mask_array = getattr(mask, group_name)
        mask_size = getattr(mask, f"{group_name}_size")
    else:
        mask_array = None
        mask_size = -1

    # Set up an iterator for us to loop over for all fields
    field_paths = group_metadata.field_paths
    field_names = group_metadata.field_names
    field_cosmologies = group_metadata.field_cosmologies
    field_units = group_metadata.field_units
    field_physicals = group_metadata.field_physicals
    field_valid_transforms = group_metadata.field_valid_transforms
    field_descriptions = group_metadata.field_descriptions
    field_compressions = group_metadata.field_compressions
    field_named_columns = group_metadata.named_columns

    dataset_iterator = zip(
        field_paths,
        field_names,
        field_cosmologies,
        field_units,
        field_physicals,
        field_valid_transforms,
        field_descriptions,
        field_compressions,
    )

    # This 'nice' piece of code ensures that our datasets have different _types_
    # for different particle types. We initially fill a dict with the properties that
    # we want, and then create a single instance of our class.

    this_dataset_bases = (__SWIFTGroupDataset, object)
    this_dataset_dict = {}

    for (
        field_path,
        field_name,
        field_cosmology,
        field_unit,
        field_physical,
        field_valid_transform,
        field_description,
        field_compression,
    ) in dataset_iterator:
        named_columns = field_named_columns[field_path]

        if named_columns is None:
            field_property = property(
                generate_getter(
                    filename,
                    field_name,
                    field_path,
                    unit=field_unit,
                    mask=mask_array,
                    mask_size=mask_size,
                    cosmo_factor=field_cosmology,
                    description=field_description,
                    compression=field_compression,
                    physical=field_physical,
                    valid_transform=field_valid_transform,
                ),
                generate_setter(field_name),
                generate_deleter(field_name),
            )
        else:
            # TODO: Handle this case with recursion.

            # Here we want to create an extra middleman object. So we can do something
            # like {ptype}.{ThisNamedColumnDataset}.column_name. This follows from the
            # above templating.

            this_named_column_dataset_bases = (__SWIFTNamedColumnDataset, object)
            this_named_column_dataset_dict = {}

            for index, column in enumerate(named_columns):
                this_named_column_dataset_dict[column] = property(
                    generate_getter(
                        filename,
                        column,
                        field_path,
                        unit=field_unit,
                        mask=mask_array,
                        mask_size=mask_size,
                        cosmo_factor=field_cosmology,
                        description=f"{field_description} [Column {index}, {column}]",
                        compression=field_compression,
                        physical=field_physical,
                        valid_transform=field_valid_transform,
                        columns=np.s_[index],
                    ),
                    generate_setter(column),
                    generate_deleter(column),
                )

            ThisNamedColumnDataset = type(
                f"{group_nice_name}{field_path.split('/')[-1]}Columns",
                this_named_column_dataset_bases,
                this_named_column_dataset_dict,
            )

            field_property = ThisNamedColumnDataset(
                field_path=field_path, named_columns=named_columns, name=field_name
            )

        this_dataset_dict[field_name] = field_property

    ThisDataset = type(
        f"{group_nice_name}Dataset", this_dataset_bases, this_dataset_dict
    )
    empty_dataset = ThisDataset(group_metadata)

    return empty_dataset


class SWIFTDataset(object):
    """
    A collection object for:

    + SWIFTUnits,
    + SWIFTMetadata,
    + SWIFTGroupDatasets

    This object, in essence, completely represents a SWIFT snapshot. You can access
    the different particles as follows:

    + SWIFTDataset.gas.particle_ids
    + SWIFTDataset.dark_matter.masses
    + SWIFTDataset.gas.smoothing_lengths

    These arrays all have units that are determined by the unit system in the file.

    The unit system is available as SWIFTDataset.units and the metadata as
    SWIFTDataset.metadata.

    Methods
    -------
    def get_units(self):
        Loads the units from the SWIFT snapshot.
    def get_metadata(self):
        Loads the metadata from the SWIFT snapshot.
    def create_particle_datasets(self):
        Creates particle datasets for whatever particle types and names
        are specified in metadata.particle_types.
    """

    def __init__(self, filename, mask=None):
        """
        Constructor for SWIFTDataset class

        Parameters
        ----------
        filename : str
            name of file containing snapshot
        mask : np.ndarray, optional
            mask object containing dataset to selected particles
        """
        self.filename = filename
        self.mask = mask

        if mask is not None:
            self.mask.convert_masks_to_ranges()

        self.get_units()
        self.get_metadata()
        self.create_datasets()

        return

    def __str__(self):
        """
        Prints out some more useful information, rather than just
        the memory location.
        """

        return f"SWIFT dataset at {self.filename}."

    def __repr__(self):
        return self.__str__()

    def get_units(self):
        """
        Loads the units from the SWIFT snapshot.

        Ordinarily this happens automatically, but you can call
        this function again if you mess things up.
        """

        self.units = SWIFTUnits(self.filename)

        return

    def get_metadata(self):
        """
        Loads the metadata from the SWIFT snapshot.

        Ordinarily this happens automatically, but you can call
        this function again if you mess things up.
        """

        self.metadata = metadata_discriminator(self.filename, self.units)

        return

    def create_datasets(self):
        """
        Creates datasets for whichever groups
        are specified in metadata.present_group_names.

        These can then be accessed using their underscore names, e.g. gas.
        """

        if not hasattr(self, "metadata"):
            self.get_metadata()

        for group_name in self.metadata.present_group_names:
            setattr(
                self,
                group_name,
                generate_datasets(
                    getattr(self.metadata, f"{group_name}_properties"), self.mask
                ),
            )

        return
