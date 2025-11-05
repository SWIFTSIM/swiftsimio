"""
Main objects for reading SWIFT datasets.

These include:

+ SWIFTUnits, which is a unit system that can be queried for units (and converts arrays
  to relevant unyt arrays when read from the HDF5 file)
+ SWIFTMetadata, which contains all of the metadata from the file
+ __SWIFTGroupDataset, which contains particle information but should never be
  directly accessed. Use ``_generate_datasets`` to create one of these. The reasoning
  here is that properties can only be added to the class afterwards, and not
  directly in an _instance_ of the class.
+ SWIFTDataset, a container class for all of the above.
"""

from swiftsimio.accelerated import read_ranges_from_file
from swiftsimio.objects import cosmo_array, cosmo_factor
from swiftsimio.masks import SWIFTMask
from swiftsimio.metadata.objects import (
    _metadata_discriminator,
    SWIFTUnits,
    SWIFTGroupMetadata,
)
from swiftsimio._handle_provider import HandleProvider

import h5py
import unyt
import numpy as np

from typing import Callable
from pathlib import Path


def _generate_getter(
    filename: str,
    name: str,
    field: str,
    unit: unyt.unyt_quantity,
    mask: None | np.ndarray,
    mask_size: int,
    cosmo_factor: cosmo_factor,
    description: str,
    compression: str,
    physical: bool,
    valid_transform: bool,
    columns: None | slice = None,
) -> Callable[["__SWIFTGroupDataset"], cosmo_array]:
    """
    Generate a function that retrieves data from file if not already in memory.

    The process is:

    a) If self._`name` exists, return it
    b) If not, open `filename`
    c) Reads filename[`field`]
    d) Set self._`name`
    e) Return self._`name`.

    Parameters
    ----------
    filename : str
        Filename of the HDF5 file that everything will be read from. Used to generate
        the HDF5 dataset.

    name : str
        Output name (snake_case) of the field.

    field : str
        Full path of field, including e.g. particle type. Examples include
        ``/PartType0/Velocities``.

    unit : unyt.unyt_quantity
        Output unit of the resultant ``cosmo_array``.

    mask : None or np.ndarray
        Mask to be used with ``accelerated.read_ranges_from_file``, i.e. an array of
        integers that describe ranges to be read from the file.

    mask_size : int
        Size of the mask if present.

    cosmo_factor : cosmo_factor
        Cosmology factor object corresponding to this array.

    description : str
        Description (read from HDF5 file) of the data.

    compression : str
        String describing the lossy compression filters that were applied to the
        data (read from the HDF5 file).

    physical : bool
        Bool that describes whether the data in the file is stored in comoving
        or physical units.

    valid_transform : bool
        Bool that describes whether converting this field from physical to comoving
        units is a valid operation.

    columns : np.lib.index_tricks.IndexEpression, optional
        Index expression corresponding to which columns to read from the numpy array.
        If not provided, we read all columns and return an n-dimensional array.

    Returns
    -------
    Callable
        A getter: callable object that gets the value of the array that has been saved to
        ``_name``. This function takes only ``self`` from the ``__SWIFTGroupDataset``
        class.

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

    def getter(self: __SWIFTGroupDataset) -> cosmo_array:
        """
        Get the data for this dataset, reading from from disk if it's not in memory.

        Parameters
        ----------
        self : __SWIFTGroupDataset
            The containing dataset class that this getter is assigned to.

        Returns
        -------
        cosmo_array
            The dataset.
        """
        current_value = getattr(self, f"_{name}")

        if current_value is not None:
            return current_value
        else:
            with h5py.File(filename, "r") as handle:
                try:
                    if mask is not None:
                        output_type = handle[field].dtype
                        output_shape = (
                            (mask_size, handle[field].shape[1])
                            if handle[field].ndim > 1 and not use_columns
                            else mask_size
                        )
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
                                # Only use column data if array is multidimensional,
                                # otherwise we will crash here
                                (
                                    handle[field][:, columns]
                                    if handle[field].ndim > 1
                                    else handle[field][:]
                                ),
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


def _generate_setter(name: str) -> Callable[["__SWIFTGroupDataset", cosmo_array], None]:
    """
    Generate a function that sets self._name to the value that is passed to it.

    Parameters
    ----------
    name : str
        The name of the attribute to set.

    Returns
    -------
    Callable
        A callable object that sets the attribute specified by ``name`` to the value
        passed to it.
    """

    def setter(self: __SWIFTGroupDataset, value: cosmo_array) -> None:
        """
        Set the (private attribute) data for this dataset.

        Parameters
        ----------
        self : __SWIFTGroupDataset
            The containing dataset class that this getter is assigned to.

        value : cosmo_array
            The data values.
        """
        setattr(self, f"_{name}", value)

        return

    return setter


def _generate_deleter(name: str) -> Callable[["__SWIFTGroupDataset"], None]:
    """
    Generate a function that destroys self._name (sets it back to None).

    Parameters
    ----------
    name : str
        The name of the field to be deleted.

    Returns
    -------
    Callable
        Callable that deletes ``name`` field.
    """

    def deleter(self: __SWIFTGroupDataset) -> None:
        """
        Delete the data for this dataset.

        Parameters
        ----------
        self : __SWIFTGroupDataset
            The containing dataset class that this getter is assigned to.
        """
        current_value = getattr(self, f"_{name}")
        del current_value
        setattr(self, f"_{name}", None)

        return

    return deleter


class __SWIFTGroupDataset(HandleProvider):
    """
    Create empty property fields.

    Do not use this class alone; it is essentially completely empty. It is filled
    with properties by `_generate_datasets`.

    On initialization we calls the generate_empty_properties function to ensure that
    defaults are set correctly.

    Parameters
    ----------
    filename : Path
        The filename to read metadata.

    group_metadata : SWIFTGroupMetadata
        The metadata used to generate empty properties.

    handle : h5py.File, optional
        File handle to read from.
    """

    filename: Path

    def __init__(
        self,
        filename: Path,
        group_metadata: SWIFTGroupMetadata,
        handle: h5py.File | None = None,
    ) -> None:
        super().__init__(handle.filename, handle=handle)
        self.units = group_metadata.units

        self.group = group_metadata.group
        self.group_name = group_metadata.group_name

        self.group_metadata = group_metadata
        self.metadata = group_metadata.metadata

        self.generate_empty_properties()
        self._close_handle_if_manager()

        return

    def generate_empty_properties(self) -> None:
        """
        Generate empty properties that will be accessed through the setters and getters.

        Initially set all of the _{name} values to None. If it doesn't _exist_ in the
        file, the variable is not created.
        """
        for field_name, field_path in zip(
            self.group_metadata.field_names, self.group_metadata.field_paths
        ):
            if field_path in self.handle:
                setattr(self, f"_{field_name}", None)
            else:
                raise AttributeError(
                    (
                        f"Cannot find attribute {field_path} in file although"
                        " it was present when searching initially."
                    )
                )

        return

    def __str__(self) -> str:
        """
        Print out available fields, not just the memory location.

        Returns
        -------
        str
            The file location and available fields.
        """
        field_names = ", ".join(self.group_metadata.field_names)
        return f"SWIFT dataset at {self.filename}. \nAvailable fields: {field_names}"

    def __repr__(self) -> str:
        """
        Print out available fields, not just the memory location.

        Returns
        -------
        str
            The file location and available fields.
        """
        return self.__str__()


class __SWIFTNamedColumnDataset(object):
    r"""
    Holder class for individual named datasets.

    Very similar to :class:`~swiftsimio.reader.__SWIFTGroupDataset` but much simpler.

    Parameters
    ----------
    field_path : str
        Path to field within hdf5 snapshot.

    named_columns : list of str
        List of categories for the variable ``name``.

    name : str
        The variable of interest.

    Examples
    --------
    For a gas particle we might be interested in the mass fractions for a number
    of elements (e.g. hydrogen, helium, carbon, etc). In a SWIFT snapshot these
    would be found in ``field_path`` = /PartType0/ElementMassFractions. The
    ``named_columns`` would be the list of elements (["hydrogen", ...]) and the
    variable we're interested in is the mass fraction ``name`` = element_mass_fraction.

    Thus,

    .. code-block:: python

        data.gas = __SWIFTNamedColumnDataset(
        "/PartType0/ElementMassFractions",
        ["hydrogen", "helium"],
        "element_mass_fraction"
        )

    would make visible:

    .. code-block:: python

        data.gas.element_mass_fraction.hydrogen
        data.gas.element_mass_fraction.helium
    """

    def __init__(self, field_path: str, named_columns: list[str], name: str) -> None:
        self.field_path = field_path
        self.named_columns = named_columns
        self.name = name

        # Need to initialise for the getter() call.
        for column in named_columns:
            setattr(self, f"_{column}", None)

        return

    def __str__(self) -> str:
        """
        Print the available column names for this dataset.

        Returns
        -------
        str
            Formatted list of column names.
        """
        return (
            f"Named columns instance with {self.named_columns} available "
            f'for "{self.name}"'
        )

    def __repr__(self) -> str:
        """
        Print the available column names for this dataset.

        Returns
        -------
        str
            Formatted list of column names.
        """
        return self.__str__()

    def __len__(self) -> int:
        """
        Get the column count.

        Returns
        -------
        int
            The number of columns.
        """
        return len(self.named_columns)

    def __eq__(self, other: "__SWIFTNamedColumnDataset") -> bool:
        """
        Check if the dataset name and available column match another's.

        Parameters
        ----------
        other : __SWIFTNamedColumnDataset
            The other dataset to compare to.

        Returns
        -------
        bool
            ``True`` if the datasets match, ``False`` otherwise.
        """
        return self.named_columns == other.named_columns and self.name == other.name


def _generate_datasets(
    filename: Path,
    group_metadata: SWIFTGroupMetadata,
    mask: SWIFTMask,
    handle: h5py.File | None = None,
) -> __SWIFTGroupDataset | __SWIFTNamedColumnDataset:
    """
    Generate a SWIFTGroupDatasets _class_ for the given particle type.

    We _must_ do the following _outside_ of the class itself, as one
    can assign properties to a _class_ but not _within_ a class
    dynamically.

    Here we loop through all of the possible properties in the metadata file.
    We then use the builtin ``property()`` function and some generators to
    create setters and getters for those properties. This will allow them
    to be accessed from outside by using SWIFTGroupDataset.name, where
    the name is, for example, coordinates.

    Parameters
    ----------
    filename : Path
        File name to read metadata.

    group_metadata : SWIFTGroupMetadata
        The metadata for the group.

    mask : SWIFTMask
        The mask object for the datasets.

    handle : h5py.File, optional
        File handle to read metadata.

    mask : SWIFTMask
        The mask object for the datasets.

    Returns
    -------
    __SWIFTGroupDataset or __SWIFTNamedColumnDataset
        The customized dataset object.
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
                _generate_getter(
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
                _generate_setter(field_name),
                _generate_deleter(field_name),
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
                    _generate_getter(
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
                    _generate_setter(column),
                    _generate_deleter(column),
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
    empty_dataset = ThisDataset(filename, group_metadata, handle=handle)

    return empty_dataset


class SWIFTDataset(HandleProvider):
    """
    A collection object for units, metadata and data objects.

    It contains:

    + ``SWIFTUnits``,
    + ``SWIFTMetadata``,
    + ``SWIFTGroupDataset``s

    This object, in essence, completely represents a SWIFT snapshot. You can access
    the different particles as follows:

    + SWIFTDataset.gas.particle_ids
    + SWIFTDataset.dark_matter.masses
    + SWIFTDataset.gas.smoothing_lengths

    These arrays all have units that are determined by the unit system in the file.

    The unit system is available as SWIFTDataset.units and the metadata as
    SWIFTDataset.metadata.

    Parameters
    ----------
    filename : str
        Name of file containing snapshot.

    mask : np.ndarray, optional
        Mask object containing dataset to selected particles.

    handle : h5py.File, optional
        File handle to read metadata.
    """

    filename: Path

    def __init__(
        self,
        filename: Path,
        mask: SWIFTMask | None = None,
        handle: h5py.File | None = None,
    ) -> None:
        super().__init__(filename, handle=handle)
        self.mask = mask
        if mask is not None:
            self.mask.convert_masks_to_ranges()

        self.get_units()
        self.get_metadata()
        self.create_datasets()

        self._close_handle_if_manager()

        return

    def __str__(self) -> str:
        """Print out some useful information, not just the memory location."""
        group_names = ", ".join(self.metadata.present_group_names)
        return f"SWIFT dataset at {self.filename}. \nAvailable groups: {group_names}"

    def __repr__(self) -> str:
        """Print out some useful information, not just the memory location."""
        return self.__str__()

    def get_units(self) -> None:
        """
        Load the units from the SWIFT snapshot.

        Ordinarily this happens automatically, but you can call
        this function again if you mess things up.
        """
        if self.mask is not None:
            # we can save ourselves the trouble of reading it again
            assert (
                self.filename == self.mask.filename
            ), f"Mask is for {self.mask.filename} but dataset is for {self.filename}."
            self.units = self.mask.units
        else:
            self.units = SWIFTUnits(self.filename, handle=self.handle)

        return

    def get_metadata(self) -> None:
        """
        Load the metadata from the SWIFT snapshot.

        Ordinarily this happens automatically, but you can call
        this function again if you mess things up.
        """
        if self.mask is not None:
            # we can save ourselves the trouble of reading it again
            assert (
                self.filename == self.mask.filename
            ), f"Mask is for {self.mask.filename} but dataset is for {self.filename}."
            self.metadata = self.mask.metadata
        else:
            self.metadata = _metadata_discriminator(
                self.filename, self.units, handle=self.handle
            )

        return

    def create_datasets(self) -> None:
        """
        Create datasets for present groups.

        Present groups are specified in metadata.present_group_names.

        These can then be accessed using their underscore names, e.g. gas.
        """
        if not hasattr(self, "metadata"):
            self.get_metadata()

        for group_name in self.metadata.present_group_names:
            setattr(
                self,
                group_name,
                _generate_datasets(
                    self.filename,
                    getattr(self.metadata, f"{group_name}_properties"),
                    self.mask,
                    handle=self.handle,
                ),
            )

        return
