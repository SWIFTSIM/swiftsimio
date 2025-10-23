"""
Handle the metadata in SWIFTsimIO files.

There is a main abstract class, ``SWIFTMetadata``, that contains the required base
methods to correctly represent the internal representation of an HDF5 file to what
SWIFTsimIO expects to be able to unpack into the object notation (e.g.
PartType0/Coordinates -> gas.coordinates).
"""

import numpy as np
import h5py
import unyt
from unyt.array import _iterable

from swiftsimio.conversions import swift_cosmology_to_astropy
from swiftsimio import metadata
from swiftsimio.objects import cosmo_array, cosmo_quantity, cosmo_factor
from swiftsimio._handle_provider import HandleProvider
from abc import ABC, abstractmethod

import re
import warnings

from datetime import datetime
from pathlib import Path


def _convert_snake_to_camel(name: str) -> str:
    """
    Place underscore between words and make all lower case.

    Parameters
    ----------
    name : str
        Name in CamelCase.

    Returns
    -------
    str
        Converted name in snake_case.
    """
    # regular expression for camel case to snake case
    # https://stackoverflow.com/a/1176023
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


class SWIFTMetadata(HandleProvider, ABC):
    """
    An abstract base class for all SWIFT-related file metadata.

    Parameters
    ----------
    filename : Path
        Filename to read metadata from.

    units : SWIFTUnits
        Units object to use.

    handle : h5py.File
        File handle to read metadata from.
    """

    # Underlying path to the file that this metadata is associated with.
    filename: Path
    # The units object associated with this file. All SWIFT metadata objects
    # must use this units system.
    units: "SWIFTUnits"
    # The header dictionary which will later be unpackaged according to the
    # metadata fields.
    header: dict
    # Whether this type of file can be masked or not (this is a fixed parameter
    # that should probably not be changed at run-time).
    masking_valid: bool = False
    # Whether this file uses shared metadata cell counts for all particle types
    # (as is the case in SOAP) or whether each type (e.g. Gas, Dark Matter, etc.)
    # has its own top-level cell grid counts.
    shared_cell_counts: str | None = None
    # Whether all the arrays in this files have the same length and order (as is
    # the case for SOAP, all arrays correspond to subhalos) or whether there are
    # multiple types (e.g. Gas, Dark Matter, etc.). Allows you to use constrain_index
    # in masking as everyone uses the same _shared mask!
    homogeneous_arrays: bool = False

    def __init__(
        self,
        filename: Path,
        units: "SWIFTUnits | None" = None,
        handle: h5py.File | None = None,
    ) -> None:
        super().__init__(filename, handle=handle)
        if units is not None:
            self.units = units
        else:
            self.units = SWIFTUnits(filename, handle=self.handle)

        # don't _close_handle_if_manager here in the ABC, let derived classes close

        return

    def load_groups(self) -> None:
        """
        Load the groups and metadata into objects.

        These are called:

            metadata.<group_name>_properties

        This contains eight arrays,

            metadata.<type>_properties.field_names
            metadata.<type>_properties.field_paths
            metadata.<type>_properties.field_units
            metadata.<type>_properties.field_cosmologies
            metadata.<type>_properties.field_descriptions
            metadata.<type>_properties.field_compressions
            metadata.<type>_properties.field_physicals
            metadata.<type>_properties.field_valid_transforms

        As well as some more information about the group.
        """
        for group, name in zip(self.present_groups, self.present_group_names):
            filetype_metadata = SWIFTGroupMetadata(
                self.filename,
                group=group,
                group_name=name,
                metadata=self,
                scale_factor=self.scale_factor,
                handle=self.handle,
            )
            setattr(self, f"{name}_properties", filetype_metadata)

        return

    def get_metadata(self) -> None:
        """Load the metadata as specified in metadata.metadata_fields."""
        for field, name in metadata.metadata_fields.metadata_fields_to_read.items():
            try:
                setattr(self, name, dict(self.handle[field].attrs))
            except KeyError:
                setattr(self, name, None)

        return

    def postprocess_header(self) -> None:
        """Do some minor postprocessing on the header to local variables."""
        # We need the scale factor to initialize `cosmo_array`s, so start with the float
        # items including the scale factor.
        # These must be unpacked as they are stored as length-1 arrays

        header_unpack_float_units = (
            metadata.metadata_fields.generate_units_header_unpack_single_float(
                mass=self.units.mass,
                length=self.units.length,
                time=self.units.time,
                current=self.units.current,
                temperature=self.units.temperature,
            )
        )
        for field, names in metadata.metadata_fields.header_unpack_single_float.items():
            try:
                if isinstance(names, list):
                    # Sometimes we store a list in case we have multiple names, for
                    # example Redshift -> metadata.redshift AND metadata.z. Can't just do
                    # the iteration because we may loop over the letters in the string.
                    for variable in names:
                        if variable in header_unpack_float_units.keys():
                            # We have an associated unit!
                            unit = header_unpack_float_units[variable]
                            setattr(
                                self,
                                variable,
                                unyt.unyt_quantity(self.header[field][0], units=unit),
                            )
                        else:
                            # No unit
                            setattr(self, variable, self.header[field][0])
                else:
                    # We can just check for the unit and set the attribute
                    variable = names
                    if variable in header_unpack_float_units.keys():
                        # We have an associated unit!
                        unit = header_unpack_float_units[variable]
                        setattr(
                            self,
                            variable,
                            unyt.unyt_quantity(self.header[field][0], units=unit),
                        )
                    else:
                        # No unit
                        setattr(self, variable, self.header[field][0])
            except KeyError:
                # Must not be present, just skip it
                continue
        # need the scale factor first for cosmology on other header attributes
        try:
            self.a = self.scale_factor
        except AttributeError:
            # These must always be present for the initialisation of cosmology properties
            self.a = 1.0
            self.scale_factor = 1.0

        # These are just read straight in to variables
        header_unpack_arrays_units = (
            metadata.metadata_fields.generate_units_header_unpack_arrays(
                mass=self.units.mass,
                length=self.units.length,
                time=self.units.time,
                current=self.units.current,
                temperature=self.units.temperature,
            )
        )
        header_unpack_arrays_cosmo_args = (
            metadata.metadata_fields.generate_cosmo_args_header_unpack_arrays(
                self.scale_factor
            )
        )

        for field, name in metadata.metadata_fields.header_unpack_arrays.items():
            try:
                if name in header_unpack_arrays_units.keys():
                    if name in header_unpack_arrays_cosmo_args.keys():
                        unpack_class = (
                            cosmo_array
                            if _iterable(self.header[field])
                            else cosmo_quantity
                        )
                        setattr(
                            self,
                            name,
                            unpack_class(
                                self.header[field],
                                units=header_unpack_arrays_units[name],
                                **header_unpack_arrays_cosmo_args[name],
                            ),
                        )
                    else:
                        setattr(
                            self,
                            name,
                            unyt.unyt_array(
                                self.header[field],
                                units=header_unpack_arrays_units[name],
                            ),
                        )
                    # This is required or we automatically get everything in CGS!
                    getattr(self, name).convert_to_units(
                        header_unpack_arrays_units[name]
                    )
                else:
                    # Must not have any units! Oh well.
                    setattr(self, name, self.header[field])
            except KeyError:
                # Must not be present, just skip it
                continue

        # Now unpack the 'mass table' type items:
        for field, name in metadata.metadata_fields.header_unpack_mass_tables.items():
            try:
                setattr(
                    self,
                    name,
                    MassTable(
                        base_mass_table=self.header[field], mass_units=self.units.mass
                    ),
                )
            except KeyError:
                setattr(
                    self,
                    name,
                    MassTable(
                        base_mass_table=np.zeros(
                            len(metadata.particle_types.particle_name_underscores)
                        ),
                        mass_units=self.units.mass,
                    ),
                )

        # These must be unpacked as 'real' strings (i.e. converted to utf-8)

        for field, name in metadata.metadata_fields.header_unpack_string.items():
            try:
                # Deal with h5py's quirkiness that fixed-sized and variable-sized
                # strings are read as strings or bytes
                # See: https://github.com/h5py/h5py/issues/2172
                raw = self.header[field]
                try:
                    string = raw.decode("utf-8")
                except AttributeError:
                    string = raw
                setattr(self, name, string)
            except KeyError:
                # Must not be present, just skip it
                setattr(self, name, "")

        # These are special cases, sorry!
        # Date and time of snapshot dump
        try:
            try:
                # Try and decode bytes, otherwise save raw string
                snapshot_date = self.header.get(
                    "SnapshotDate", self.header.get("Snapshot date", b"")
                ).decode("utf-8")
            except AttributeError:
                snapshot_date = self.header.get(
                    "SnapshotDate", self.header.get("Snapshot date", "")
                )
            try:
                self.snapshot_date = datetime.strptime(
                    snapshot_date, "%H:%M:%S %Y-%m-%d %Z"
                )
            except ValueError:
                # Backwards compatibility; this was used previously due to simplicity
                # but is not portable between regions. So if you ran a simulation on
                # a British (en_GB) machine, and then tried to read on a Dutch
                # machine (nl_NL), this would _not_ work because %c is different.
                try:
                    self.snapshot_date = datetime.strptime(snapshot_date, "%c\n")
                except ValueError:
                    # Oh dear this has gone _very_wrong. Let's just keep it as a string.
                    self.snapshot_date = snapshot_date
        except KeyError:
            # Old file
            pass

        # get photon group edges RT dataset from the SubgridScheme group
        try:
            self.photon_group_edges = (
                self.handle["SubgridScheme/PhotonGroupEdges"][:] / self.units.time
            )
        except KeyError:
            self.photon_group_edges = None

        # get reduced speed of light RT dataset from the SubgridScheme group
        try:
            self.reduced_lightspeed = (
                self.handle["SubgridScheme/ReducedLightspeed"][0]
                * self.units.length
                / self.units.time
            )
        except KeyError:
            self.reduced_lightspeed = None

        # Store these separately as self.n_gas = number of gas particles for example
        for part_number, (_, part_name) in enumerate(
            metadata.particle_types.particle_name_underscores.items()
        ):
            try:
                setattr(self, f"n_{part_name}", self.num_part[part_number])
            except IndexError:
                # Backwards compatibility; mass/number table can change size.
                setattr(self, f"n_{part_name}", 0)

        # Need to unpack the gas gamma for cosmology
        try:
            self.gas_gamma = self.hydro_scheme["Adiabatic index"]
        except (KeyError, TypeError):
            # We can set a default and print a message whenever we require this value
            self.gas_gamma = None

        return

    def extract_cosmology(self) -> None:
        """
        Create an astropy.cosmology object from the internal cosmology system.

        This will be saved as ``self.cosmology``.
        """
        if self.cosmology_raw is not None:
            cosmo = self.cosmology_raw
        else:
            cosmo = {"Cosmological run": 0}

        if cosmo.get("Cosmological run", 0):
            self.cosmology = swift_cosmology_to_astropy(cosmo, units=self.units)
        else:
            self.cosmology = None

        return

    @property
    @abstractmethod
    def present_groups(self) -> list[str]:
        """
        Get the present groups.

        A property giving the present particle groups in the file to be unpackaged
        into top-level properties. For instance, in a regular snapshot, this would be
        ["PartType0", "PartType1", "PartType4", ...]. In SOAP, this would be
        ["SO/200_crit", "SO/200_mean", ...], i.e. one per aperture.

        Returns
        -------
        list[str]
            The list of present groups.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def present_group_names(self) -> list[str]:
        """
        Get the present group names.

        A property giving the mapping for the names in ``present_groups`` to what the
        objects are called on the SWIFTsimIO objects. For instance, in a regular snapshot,
        this would be ["gas", "dark_matter", "stars", ...]. In SOAP, this would be
        ["spherical_overdensity_200_crit", ...].

        Returns
        -------
        list[str]
            The list of present group names.
        """
        raise NotImplementedError

    @property
    def partial_snapshot(self) -> bool:
        """
        Check if this is a partial (e.g. a ``x.0.hdf5`` file).

        Returns
        -------
        bool
            ``True`` if the file is a partial file, else ``False``.
        """
        return False

    @staticmethod
    @abstractmethod
    def get_nice_name(group: str) -> str:
        """
        Convert the group name to a user-readable name.

        Parameters
        ----------
        group : str
            The group name as used in the hdf5 file.

        Returns
        -------
        str
            The user-readable version of the name.
        """
        raise NotImplementedError


class MassTable(object):
    """
    Extract a mass table to local variables based on the particle type names.

    Parameters
    ----------
    base_mass_table : np.array
        Mass table of the same length as the number of particle types.

    mass_units : unyt_quantity
        Base mass units for the simulation.
    """

    def __init__(
        self, base_mass_table: np.array, mass_units: unyt.unyt_quantity
    ) -> None:
        # TODO: Extract these names from the files themselves if possible.

        for index, name in enumerate(
            metadata.particle_types.particle_name_underscores.values()
        ):
            try:
                setattr(
                    self,
                    name,
                    unyt.unyt_quantity(base_mass_table[index], units=mass_units),
                )
            except IndexError:
                # Backwards compatible.
                setattr(self, name, None)

        return

    def __str__(self) -> str:
        """
        Print a description of the mass table.

        Returns
        -------
        str
            The mass table description.
        """
        return (
            "Mass table for "
            f"{' '.join(metadata.particle_types.particle_name_underscores.values())}"
        )

    def __repr__(self) -> str:
        """
        Print a description of the mass table.

        Returns
        -------
        str
            The mass table description.
        """
        return self.__str__()


class MappingTable(object):
    """
    Provide a table mapping from one named column instance to the other.

    Initially designed for the mapping between dust and elements.

    Parameters
    ----------
    data : np.ndarray
        The data array providing the mapping between the named
        columns. Should be of size N x M, where N is the number
        of elements in ``named_columns_x`` and M the number
        of elements in ``named_columns_y``.

    named_columns_x : list[str]
        The names of the columns in the first axis.

    named_columns_y : list[str]
        The names of the columns in the second axis.

    named_columns_x_name : str
        The name of the first mapping.

    named_columns_y_name : str
        The name of the second mapping.
    """

    def __init__(
        self,
        data: np.ndarray,
        named_columns_x: list[str],
        named_columns_y: list[str],
        named_columns_x_name: str,
        named_columns_y_name: str,
    ) -> None:
        self.data = data
        self.named_columns_x = named_columns_x
        self.named_columns_y = named_columns_y
        self.named_columns_x_name = named_columns_x_name
        self.named_columns_y_name = named_columns_y_name

        for x, name_x in enumerate(named_columns_x):
            for y, name_y in enumerate(named_columns_y):
                setattr(self, f"{name_x.lower()}_to_{name_y.lower()}", data[x][y])

        return

    def __str__(self) -> str:
        """
        Print a description of the mapping table.

        Returns
        -------
        str
            The mapping table description.
        """
        return (
            f"Mapping table from {self.named_columns_x_name} to "
            f"{self.named_columns_y_name}, containing {len(self.data)} "
            f"by {len(self.data[0])} elements."
        )

    def __repr__(self) -> str:
        """
        Print a description of the mapping table.

        Returns
        -------
        str
            The mapping table description.
        """
        return f"{self.__str__()}. Raw data: \n{self.data}."


class SWIFTGroupMetadata(HandleProvider):
    """
    Provide the metadata for one hdf5 Group.

    This, for instance, could be ``PartType0``, or ``gas``. This will load in
    the names of all datasets, their units, possible named fields,
    and their cosmology, and present them for use in the actual i/o routines.

    Parameters
    ----------
    filename : Path
        Filename to read metadata from.

    group : str
        The name of the group in the hdf5 file.

    group_name : str
        The corresponding group name for swiftsimio.

    metadata : SWIFTMetadata
        The snapshot metadata.

    scale_factor : float
        The snapshot scale factor.

    handle : h5py.File
        File handle to read metadata from.
    """

    filename: Path

    def __init__(
        self,
        filename: Path,
        group: str,
        group_name: str,
        metadata: "SWIFTMetadata",
        scale_factor: float,
        handle: h5py.File,
    ) -> None:
        super().__init__(filename, handle=handle)
        self.group = group
        self.group_name = group_name
        self.metadata = metadata
        self.units = metadata.units
        self.scale_factor = scale_factor

        self.load_metadata()

        self._close_handle_if_manager()

        return

    def __str__(self) -> str:
        """
        Print a description of the metadata object.

        Returns
        -------
        str
            The description.
        """
        return f"Metadata class for {self.group} ({self.group_name})"

    def __repr__(self) -> str:
        """
        Print a description of the metadata object.

        Returns
        -------
        str
            The description.
        """
        return self.__str__()

    def load_metadata(self) -> None:
        """
        Load the required metadata.

        This includes loading the field names, units and descriptions, as well as the
        cosmology metadata and any custom named columns.
        """
        self.load_field_names()
        self.load_field_units()
        self.load_field_descriptions()
        self.load_field_compressions()
        self.load_cosmology()
        self.load_physical()
        self.load_valid_transforms()
        self.load_named_columns()

        return

    def load_field_names(self) -> None:
        """Load in only the field names."""
        # Skip fields which are groups themselves
        self.field_paths = []
        self.field_names = []
        for item in self.handle[f"{self.group}"].keys():
            # Skip fields which are groups themselves
            if f"{self.group}/{item}" not in self.metadata.present_groups:
                self.field_paths.append(f"{self.group}/{item}")
                self.field_names.append(_convert_snake_to_camel(item))

        return

    def load_field_units(self) -> None:
        """Load in the units from each dataset."""
        unit_dict = {
            "I": self.units.current,
            "L": self.units.length,
            "M": self.units.mass,
            "T": self.units.temperature,
            "t": self.units.time,
        }

        def get_units(unit_attribute: h5py.AttributeManager) -> unyt.Unit | None:
            """
            Get the units from the HDF5 attributes.

            Parameters
            ----------
            unit_attribute : h5py.AttributeManager
                The attribute containing unit metadata.

            Returns
            -------
            unyt.Unit or None
                The loaded units.
            """
            units = 1.0

            for exponent, unit in unit_dict.items():
                # We store the 'unit exponent' in the SWIFT metadata. This corresponds
                # to the power we need to raise each unit to, to return the correct units
                try:
                    # Check if the exponent is 0 manually because of float precision
                    unit_exponent = unit_attribute[f"U_{exponent} exponent"][0]
                    if unit_exponent != 0.0:
                        units *= unit**unit_exponent
                except KeyError:
                    # Can't load that data!
                    # We should probably warn the user here...
                    pass

            # Deal with case where we _really_ have a dimensionless quantity.
            if not hasattr(units, "units"):
                units = None

            return units

        self.field_units = [get_units(self.handle[x].attrs) for x in self.field_paths]

        return

    def load_field_descriptions(self) -> None:
        """
        Load in the text descriptions of the fields for each dataset.

        For SOAP filetypes a description of the mask is included.
        """

        def get_desc(dataset: h5py.Dataset) -> str:
            """
            Get the description metadata.

            Parameters
            ----------
            dataset : h5py.Dataset
                The dataset for which to get the description.

            Returns
            -------
            str
                The description of the dataset.
            """
            try:
                description = dataset.attrs["Description"].decode("utf-8")
            except AttributeError:
                # Description is saved as a string not bytes
                description = dataset.attrs["Description"]
            except KeyError:
                # Can't load description!
                description = "No description available"

            is_masked = dataset.attrs.get("Masked", False)
            if not is_masked:
                return description + " Not masked."

            mask_datasets = dataset.attrs["Mask Datasets"]
            mask_threshold = dataset.attrs["Mask Threshold"]
            if len(mask_datasets) == 1:
                mask_str = (
                    " Only computed for objects with "
                    f"{mask_datasets[0]} >= {mask_threshold}."
                )
            else:
                mask_str = (
                    " Only computed for objects where "
                    f"{' + '.join(mask_datasets)} >= {mask_threshold}."
                )
            return description + mask_str

        self.field_descriptions = [get_desc(self.handle[x]) for x in self.field_paths]

        return

    def load_field_compressions(self) -> None:
        """Load in the string describing the compression filters for each dataset."""

        def get_comp(dataset: h5py.Dataset) -> str:
            """
            Load the compression metadata for a dataset.

            Parameters
            ----------
            dataset : h5py.Dataset
                The dataset to load the compression metadata for.

            Returns
            -------
            str
                The compression metadata.
            """
            try:
                # SOAP catalogues can be compressed/uncompressed
                is_compressed = dataset.attrs["Is Compressed"]
            except KeyError:
                is_compressed = True
            try:
                comp = dataset.attrs["Lossy compression filter"].decode("utf-8")
            except AttributeError:
                # Compression is saved as str not bytes
                comp = dataset.attrs["Lossy compression filter"]
            except KeyError:
                # Can't load compression string!
                comp = "No compression info available"

            return comp if is_compressed else "Not compressed."

        self.field_compressions = [get_comp(self.handle[x]) for x in self.field_paths]

        return

    def load_cosmology(self) -> None:
        """Load in the field cosmologies."""
        current_scale_factor = self.scale_factor

        def get_cosmo(dataset: str) -> cosmo_factor:
            """
            Generate a cosmo factor from the metadata.

            Parameters
            ----------
            dataset : str
                The name of the dataset to read metadata for.

            Returns
            -------
            cosmo_factor
                A :class:`~swiftsimio.objects.cosmo_factor` setup from the metadata.
            """
            try:
                cosmo_exponent = dataset.attrs["a-scale exponent"][0]
            except KeyError:
                # Can't load, 'graceful' fallback.
                cosmo_exponent = 0.0

            return cosmo_factor.create(current_scale_factor, cosmo_exponent)

        self.field_cosmologies = [get_cosmo(self.handle[x]) for x in self.field_paths]

        return

    def load_physical(self) -> None:
        """Load in whether the field is saved as comoving or physical."""

        def get_physical(dataset: h5py.Dataset) -> bool:
            """
            Check if the metadata item is stored in physical units.

            Parameters
            ----------
            dataset : h5py.Dataset
                The dataset to be checked.

            Returns
            -------
            bool
                ``True`` if stored in physical units, else ``False``.
            """
            try:
                physical = dataset.attrs["Value stored as physical"][0] == 1
            except KeyError:
                physical = False
            return physical

        self.field_physicals = [get_physical(self.handle[x]) for x in self.field_paths]

        return

    def load_valid_transforms(self) -> None:
        """Load in whether the field can be converted to comoving."""

        def get_valid_transform(dataset: h5py.Dataset) -> bool:
            """
            Retrieve the metadata giving whether comoving/physical conversion is allowed.

            Parameters
            ----------
            dataset : str
                The dataset to be checked.

            Returns
            -------
            bool
                ``True`` if conversion is allowed or metadata absent (old file), ``False``
                otherwise.
            """
            try:
                valid_transform = (
                    dataset.attrs["Property can be converted to comoving"][0] == 1
                )
            except KeyError:
                # For backward compatibility default to True
                valid_transform = True
            return valid_transform

        self.field_valid_transforms = [
            get_valid_transform(self.handle[x]) for x in self.field_paths
        ]

        return

    def load_named_columns(self) -> None:
        """Load the named column data for relevant fields."""
        named_columns = {}

        for field in self.field_paths:
            property_name = field.split("/")[-1]

            # Not all datasets have named columns
            named_columns_metadata = getattr(self.metadata, "named_columns", {})

            if property_name in named_columns_metadata.keys():
                field_names = self.metadata.named_columns[property_name]

                # Now need to make a decision on capitalisation. If we have a set of
                # words with only one capital in them, then it's likely that they are
                # element names or something similar, so they should be lower case.
                # If on average we have many more capitals, then they are likely to be
                # ionized fractions (e.g. HeII) and so we want to leave them with their
                # original capitalisation.

                def num_capitals(s: str) -> int:
                    """
                    Count the number of upper case letters in the string.

                    Parameters
                    ----------
                    s : str
                        The string to analyse.

                    Returns
                    -------
                    int
                        The number of upper case letters in the string.
                    """
                    return sum(1 for c in s if c.isupper())

                mean_num_capitals = sum(map(num_capitals, field_names)) / len(
                    field_names
                )

                if mean_num_capitals < 1.01:
                    # Decapitalise them as they are likely individual element names
                    formatted_field_names = [x.lower() for x in field_names]
                else:
                    formatted_field_names = field_names

                named_columns[field] = formatted_field_names
            else:
                named_columns[field] = None

        self.named_columns = named_columns

        return


class SWIFTUnits(HandleProvider):
    """
    Generate a :mod:`unyt` system that can be used with SWIFT data.

    These give the unit mass, length, time, current, and temperature as
    unyt unit variables in simulation units. I.e. you can take any value
    that you get out of the code and multiply it by the appropriate values
    to get it 'unyt-ified' with the correct units.

    Parameters
    ----------
    filename : Path
        Name of file to read units from.

    handle : h5py.File, optional
        The h5py file handle, optional. Will open a new handle with the
        filename if required.
    """

    mass: unyt.unyt_quantity
    length: unyt.unyt_quantity
    time: unyt.unyt_quantity
    current: unyt.unyt_quantity
    temperature: unyt.unyt_quantity

    def __init__(self, filename: Path, handle: h5py.File | None = None) -> None:
        super().__init__(filename, handle=handle)
        self.get_unit_dictionary()

        self._close_handle_if_manager()

        return

    def get_unit_dictionary(self) -> None:
        """
        Store unit data and metadata.

        Length 1 arrays are used to store the unit data. This dictionary
        also contains the metadata information that connects the unyt
        objects to the names that are stored in the SWIFT snapshots.
        """
        self.units = {
            name: unyt.unyt_quantity(
                value[0], units=metadata.unit_types.unit_names_to_unyt[name]
            )
            for name, value in self.handle["Units"].attrs.items()
        }

        # We now unpack this into variables.
        self.mass = metadata.unit_types.find_nearest_base_unit(
            self.units["Unit mass in cgs (U_M)"], "mass"
        )
        self.length = metadata.unit_types.find_nearest_base_unit(
            self.units["Unit length in cgs (U_L)"], "length"
        )
        self.time = metadata.unit_types.find_nearest_base_unit(
            self.units["Unit time in cgs (U_t)"], "time"
        )
        self.current = metadata.unit_types.find_nearest_base_unit(
            self.units["Unit current in cgs (U_I)"], "current"
        )
        self.temperature = metadata.unit_types.find_nearest_base_unit(
            self.units["Unit temperature in cgs (U_T)"], "temperature"
        )


def _metadata_discriminator(
    filename: Path, units: SWIFTUnits, handle: h5py.File | None = None
) -> "SWIFTMetadata":
    """
    Determine the type of metadata object to construct.

    Parameters
    ----------
    filename : Path
        Name of the file to read metadata from.

    units : SWIFTUnits
        The units object associated with the file.

    handle : h5py.File, optional
        File handle to read metadata from.

    Returns
    -------
    SWIFTMetadata
        The appropriate metadata object for the file type.
    """
    # Old snapshots did not have OutputType, so we need to default to FullVolume
    if handle is None:
        with h5py.File(filename, "r") as local_handle:
            file_type = local_handle["Header"].attrs.get("OutputType", "FullVolume")
    else:
        file_type = handle["Header"].attrs.get("OutputType", "FullVolume")

    if isinstance(file_type, bytes):
        file_type = file_type.decode("utf-8")

    if file_type in ["FullVolume"]:
        return SWIFTSnapshotMetadata(filename, units, handle=handle)
    elif file_type in ["SOAP"]:
        return SWIFTSOAPMetadata(filename, units, handle=handle)
    elif file_type in ["FOF"]:
        return SWIFTFOFMetadata(filename, units, handle=handle)
    else:
        raise ValueError(f"File type {file_type} not recognised.")


class SWIFTSnapshotMetadata(SWIFTMetadata):
    """
    Provide a metadata interface for SWIFT snapshot files.

    For more documentation see :class:`~swiftsimio.metadata.objects.SWIFTMetadata`.

    Parameters
    ----------
    filename : Path
        Filename to read metadata from.

    units : SWIFTUnits
        Units object to use.

    handle : h5py.File, optional
        File handle to read from.
    """

    masking_valid: bool = True

    def __init__(
        self,
        filename: Path,
        units: SWIFTUnits | None = None,
        handle: h5py.File | None = None,
    ) -> None:
        super().__init__(filename, units=units, handle=handle)
        self.get_metadata()
        self.get_named_column_metadata()
        self.get_mapping_metadata()

        self.postprocess_header()

        self.load_groups()
        self.extract_cosmology()

        self._close_handle_if_manager()

        return

    def get_named_column_metadata(self) -> None:
        """
        Load the custom named column metadata from SubgridScheme/NamedColumns.

        If name column didn't exist just set an empty :obj:`dict` instead.
        """
        try:
            data = self.handle["SubgridScheme/NamedColumns"]

            self.named_columns = {
                k: [x.decode("utf-8") for x in data[k][:]] for k in data.keys()
            }
        except KeyError:
            self.named_columns = {}

        return

    def get_mapping_metadata(self) -> None:
        """
        Get the mappings based on the named columns (must have already been read).

        From the form:

        SubgridScheme/{X}To{Y}Mapping.

        Includes a hack of `Dust` -> `Grains` that will be deprecated.
        """
        try:
            possible_keys = self.handle["SubgridScheme"].keys()

            available_keys = [key for key in possible_keys if key.endswith("Mapping")]
            available_data = [
                self.handle[f"SubgridScheme/{key}"][:] for key in available_keys
            ]
        except KeyError:
            available_keys = []
            available_data = []

        # Keys have form {X}To{Y}Mapping
        regex = r"([a-zA-Z]*)To([a-zA-Z]*)Mapping"
        compiled = re.compile(regex)

        for key, data in zip(available_keys, available_data):
            matched = compiled.matched(key)
            snake_case = _convert_snake_to_camel(key)

            if matched:
                x = matched.group(1)
                y = matched.group(2)

                if x == "Grain":
                    warnings.warn(
                        "Use of the GrainToElementMapping is deprecated, please use a "
                        "newer version of SWIFT to run this simulation.",
                        DeprecationWarning,
                    )

                    x = "Dust"

                named_column_name_x = [
                    key for key in self.named_columns.keys() if key.startswith(x)
                ][0]
                named_column_name_y = [
                    key for key in self.named_columns.keys() if key.startswith(y)
                ][0]

                setattr(
                    self,
                    snake_case,
                    MappingTable(
                        data=data,
                        named_columns_x=self.named_columns[named_column_name_x],
                        named_columns_y=self.named_columns[named_column_name_y],
                        named_columns_x_name=named_column_name_x,
                        named_columns_y_name=named_column_name_y,
                    ),
                )

        return

    @property
    def present_groups(self) -> list[str]:
        """
        Get the groups containing datasets that are present in the file.

        Returns
        -------
        list[str]
            List of present groups.
        """
        types = np.where(np.array(getattr(self, "has_type", self.num_part)) != 0)[0]
        return [f"PartType{i}" for i in types]

    @property
    def present_group_names(self) -> list[str]:
        """
        Get the names of the groups that we want to expose.

        Returns
        -------
        list[str]
            List of names to expose.
        """
        return [
            metadata.particle_types.particle_name_underscores[x]
            for x in self.present_groups
        ]

    @property
    def code_info(self) -> str:
        """
        Get and format a nicely printed set of code information.

        Formatting is as:

        Name (Git Branch)
        Git Revision
        Git Date

        Returns
        -------
        str
            The code information.
        """

        def format_string(param: str) -> str:
            """
            Fetch a string value from metadata and decode.

            Parameters
            ----------
            param : str
                The name of the field to retrieve from the ``code`` metadata.

            Returns
            -------
            str
                The decoded string value.
            """
            return self.code[param].decode("utf-8")

        output = (
            f"{format_string('Code')} ({format_string('Git Branch')})\n"
            f"{format_string('Git Revision')}\n"
            f"{format_string('Git Date')}"
        )

        return output

    @property
    def compiler_info(self) -> str:
        """
        Get and format information about the compiler.

        Formatting is as:

        Compiler Name (Compiler Version)
        MPI library

        Returns
        -------
        str
            The compiler information.
        """

        def format_string(param: str) -> str:
            """
            Fetch a string value from metadata, decode and format.

            Parameters
            ----------
            param : str
                The name of the field to retrieve from the ``code`` metadata.

            Returns
            -------
            str
                The decoded string value.
            """
            return self.code[param].decode("utf-8")

        output = (
            f"{format_string('Compiler Name')} ({format_string('Compiler Version')})\n"
            f"{format_string('MPI library')}"
        )

        return output

    @property
    def library_info(self) -> str:
        """
        Get and format information about the libraries used.

        Formatting is as:

        FFTW vFFTW library version
        GSL vGSL library version
        HDF5 vHDF5 library version

        Returns
        -------
        str
            The library information.
        """

        def format_string(param: str) -> str:
            """
            Fetch a string value from metadata, decode and format.

            Parameters
            ----------
            param : str
                The name of the field to retrieve from the ``code`` metadata.

            Returns
            -------
            str
                The decoded string value.
            """
            return self.code[f"{param} library version"].decode("utf-8")

        output = (
            f"FFTW v{format_string('FFTW')}\n"
            f"GSL v{format_string('GSL')}\n"
            f"HDF5 v{format_string('HDF5')}"
        )

        return output

    @property
    def hydro_info(self) -> str:
        r"""
        Get and format information about the hydro scheme.

        Formatting is as:

        Scheme
        Kernel function in DimensionD
        $\eta$ = Kernel eta (Kernel target N_ngb $N_{ngb}$)
        $C_{\rm CFL}$ = CFL parameter

        Returns
        -------
        str
            Hydro scheme information.
        """

        def format_float(param: str) -> str:
            """
            Fetch a float value from metadata and format.

            Parameters
            ----------
            param : str
                The name of the field to retrieve from the ``hydro_scheme`` metadata.

            Returns
            -------
            str
                The float value formatted to 2 decimal places.
            """
            return f"{self.hydro_scheme[param][0]:4.2f}"

        def get_int(param: str) -> int:
            """
            Fetch an integer value from the metadata.

            Parameters
            ----------
            param : int
                The name of the field to retrieve from the ``hydro_scheme`` metadata.

            Returns
            -------
            str
                The integer value.
            """
            return int(self.hydro_scheme[param][0])

        def format_string(param: str) -> str:
            """
            Fetch a string value from metadata and decode.

            Parameters
            ----------
            param : str
                The name of the field to retrieve from the ``hydro_scheme`` metadata.

            Returns
            -------
            str
                The decoded string value.
            """
            return self.hydro_scheme[param].decode("utf-8")

        output = (
            f"{format_string('Scheme')}\n"
            f"{format_string('Kernel function')} in {get_int('Dimension')}D\n"
            rf"$\eta$ = {format_float('Kernel eta')} "
            rf"({format_float('Kernel target N_ngb')} $N_{{ngb}}$)"
            "\n"
            rf"$C_{{\rm CFL}}$ = {format_float('CFL parameter')}"
        )

        return output

    @property
    def viscosity_info(self) -> str:
        r"""
        Get and format information about the viscosity scheme.

        Formatting is as:

        Viscosity Model
        $\alpha_{V, 0}$ = Alpha viscosity, $\ell_V$ = Viscosity decay length \
        [internal units], $\beta_V$ = Beta viscosity
        Alpha viscosity (min) < $\alpha_V$ < Alpha viscosity (max)

        Returns
        -------
        str
            Viscosity scheme information.
        """

        def format_float(param: str) -> str:
            """
            Fetch a float value from metadata and format.

            Parameters
            ----------
            param : str
                The name of the field to retrieve from the ``hydro_scheme`` metadata.

            Returns
            -------
            str
                The float value formatted to 2 decimal places.
            """
            return f"{self.hydro_scheme[param][0]:4.2f}"

        def format_string(param: str) -> str:
            """
            Fetch a string value from metadata and decode.

            Parameters
            ----------
            param : str
                The name of the field to retrieve from the ``hydro_scheme`` metadata.

            Returns
            -------
            str
                The decoded string value.
            """
            return self.hydro_scheme[param].decode("utf-8")

        output = (
            f"{format_string('Viscosity Model')}\n"
            rf"$\alpha_{{V, 0}}$ = {format_float('Alpha viscosity')}, "
            rf"$\ell_V$ = {format_float('Viscosity decay length [internal units]')}, "
            rf"$\beta_V$ = {format_float('Beta viscosity')}"
            "\n"
            rf"{format_float('Alpha viscosity (min)')} < $\alpha_V$ < "
            rf"{format_float('Alpha viscosity (max)')}"
        )

        return output

    @property
    def diffusion_info(self) -> str:
        r"""
        Get and format information about the diffusion scheme.

        Formatting is as:

        $\alpha_{D, 0}$ = Diffusion alpha, $\beta_D$ = Diffusion beta
        Diffusion alpha (min) < $\alpha_D$ < Diffusion alpha (max)

        Returns
        -------
        str
            Formatted diffusion scheme information.
        """

        def format_float(param: str) -> str:
            """
            Fetch a float value from metadata and format.

            Parameters
            ----------
            param : str
                The name of the field to retrieve from the ``hydro_scheme`` metadata.

            Returns
            -------
            str
                The float value formatted to 2 decimal places.
            """
            return f"{self.hydro_scheme[param][0]:4.2f}"

        output = (
            rf"$\alpha_{{D, 0}}$ = {format_float('Diffusion alpha')}, "
            rf"$\beta_D$ = {format_float('Diffusion beta')}"
            "\n"
            rf"${format_float('Diffusion alpha (min)')} < "
            rf"\alpha_D < {format_float('Diffusion alpha (max)')}$"
        )

        return output

    @property
    def partial_snapshot(self) -> bool:
        """
        Check if this is a partial (e.g. a "x.0.hdf5" file).

        Returns
        -------
        bool
            ``True`` if the file is a partial file, else ``False``.
        """
        # Partial snapshots have num_files_per_snapshot set to 1. Virtual snapshots
        # collating multiple sub-snapshots together have num_files_per_snapshot = 1.

        return self.num_files_per_snapshot > 1

    @staticmethod
    def get_nice_name(group: str) -> str:
        """
        Convert the group name to a user-readable name.

        Parameters
        ----------
        group : str
            The group name as used in the hdf5 file.

        Returns
        -------
        str
            The user-readable version of the name.
        """
        return metadata.particle_types.particle_name_class[group]


class SWIFTFOFMetadata(SWIFTMetadata):
    """
    Provide a metadata interface for FOF catalogue files.

    For more documentation see :class:`~swiftsimio.metadata.objects.SWIFTMetadata`.

    Parameters
    ----------
    filename : Path
        Filename to read metadata from.

    units : SWIFTUnits
        Units object to use.

    handle : h5py.File, optional
        File handle to read from.
    """

    homogeneous_arrays: bool = True

    def __init__(
        self,
        filename: Path,
        units: SWIFTUnits | None = None,
        handle: h5py.File | None = None,
    ) -> None:
        super().__init__(filename, units=units, handle=handle)
        self.get_metadata()
        self.postprocess_header()

        self.load_groups()
        self.extract_cosmology()

        self._close_handle_if_manager()

        return

    @property
    def present_groups(self) -> list[str]:
        """
        The groups containing datasets that are present in the file.

        Returns
        -------
        list[str]
            List of available subhalo types.
        """
        return ["Groups"]

    @property
    def present_group_names(self) -> list[str]:
        """
        Provide the names of the groups that we want to expose.

        Returns
        -------
        list[str]
            List of the available groups.
        """
        return ["fof_groups"]

    @staticmethod
    def get_nice_name(group: str) -> str:
        """
        Convert the group name to a user-readable name.

        Parameters
        ----------
        group : str
            The group name as used in the hdf5 file.

        Returns
        -------
        str
            The user-readable version of the name.
        """
        return "FOFGroups"


class SWIFTSOAPMetadata(SWIFTMetadata):
    """
    Provide a metadata interface for SOAP catalogue files.

    For more documentation see :class:`~swiftsimio.metadata.objects.SWIFTMetadata`.

    Parameters
    ----------
    filename : Path
        Filename to read metadata from.

    units : SWIFTUnits
        Units object to use.

    handle : h5py.File, optional
        File handle to read from.
    """

    masking_valid: bool = True
    shared_cell_counts: str = "Subhalos"
    homogeneous_arrays: bool = True

    def __init__(
        self,
        filename: Path,
        units: SWIFTUnits | None = None,
        handle: h5py.File | None = None,
    ) -> None:
        super().__init__(filename, units=units, handle=handle)
        self.get_metadata()
        self.postprocess_header()
        self.unpack_subhalo_number()

        self.load_groups()
        self.extract_cosmology()

        self._close_handle_if_manager()

        return

    def unpack_subhalo_number(self) -> None:
        """Set the subhalo count."""
        self.n_subhalos = int(self.num_subhalo[0])
        return

    @property
    def present_groups(self) -> list[str]:
        """
        The groups containing datasets that are present in the file.

        Returns
        -------
        list[str]
            List of available subhalo types.
        """
        return self.subhalo_types

    @property
    def present_group_names(self) -> list[str]:
        """
        Provide the names of the groups that we want to expose.

        Returns
        -------
        list[str]
            List of the available groups.
        """
        return [
            metadata.soap_types.get_soap_name_underscore(x) for x in self.present_groups
        ]

    @staticmethod
    def get_nice_name(group: str) -> str:
        """
        Get the de-acronymed name of a specified group.

        Parameters
        ----------
        group : str
            The name as it appears in the SOAP file.

        Returns
        -------
        str
            The de-acronymed name.
        """
        return metadata.soap_types.get_soap_name_nice(group)
