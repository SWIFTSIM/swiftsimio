"""
Objects describing the metadata in SWIFTsimIO files. There is a main
abstract class, ``SWIFTMetadata``, that contains the required base
methods to correctly represent the internal representation of an
HDF5 file to what SWIFTsimIO expects to be able to unpack into the
object notation (e.g. PartType0/Coordinates -> gas.coordinates).
"""

import numpy as np
import unyt

import h5py
from swiftsimio.conversions import swift_cosmology_to_astropy
from swiftsimio import metadata
from swiftsimio.objects import cosmo_array, cosmo_factor, a
from abc import ABC, abstractmethod

import re
import warnings

from datetime import datetime
from pathlib import Path

from typing import List, Optional


class SWIFTMetadata(ABC):
    """
    An abstract base class for all SWIFT-related file metadata.
    """

    # Underlying path to the file that this metadata is associated with.
    filename: str
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

    @abstractmethod
    def __init__(self, filename, units: "SWIFTUnits"):
        raise NotImplementedError

    @property
    def handle(self):
        # Handle, which is shared with units. Units handles
        # file opening and closing.
        return self.units.handle

    def load_groups(self):
        """
        Loads the groups and metadata into objects:

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
                group=group,
                group_name=name,
                metadata=self,
                scale_factor=self.scale_factor,
            )
            setattr(self, f"{name}_properties", filetype_metadata)

        return

    def get_metadata(self):
        """
        Loads the metadata as specified in metadata.metadata_fields.
        """

        for field, name in metadata.metadata_fields.metadata_fields_to_read.items():
            try:
                setattr(self, name, dict(self.handle[field].attrs))
            except KeyError:
                setattr(self, name, None)

        return

    def postprocess_header(self):
        """
        Some minor postprocessing on the header to local variables.
        """

        # We need the scale factor to initialize `cosmo_array`s, so start with the float
        # items including the scale factor.
        # These must be unpacked as they are stored as length-1 arrays

        header_unpack_float_units = (
            metadata.metadata_fields.generate_units_header_unpack_single_float(
                m=self.units.mass,
                l=self.units.length,
                t=self.units.time,
                I=self.units.current,
                T=self.units.temperature,
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
                m=self.units.mass,
                l=self.units.length,
                t=self.units.time,
                I=self.units.current,
                T=self.units.temperature,
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
                        setattr(
                            self,
                            name,
                            cosmo_array(
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

    def extract_cosmology(self):
        """
        Creates an astropy.cosmology object from the internal cosmology system.

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
        A property giving the present particle groups in the file to be unpackaged
        into top-level properties. For instance, in a regular snapshot, this would be
        ["PartType0", "PartType1", "PartType4", ...]. In SOAP, this would be
        ["SO/200_crit", "SO/200_mean", ...], i.e. one per aperture.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def present_group_names(self) -> list[str]:
        """
        A property giving the mapping for the names in ``present_groups`` to what the
        objects are called on the SWIFTsimIO objects. For instance, in a regular snapshot,
        this would be ["gas", "dark_matter", "stars", ...]. In SOAP, this would be
        ["spherical_overdensity_200_crit", ...].
        """
        raise NotImplementedError

    @property
    def partial_snapshot(self) -> bool:
        """
        A property defining whether this is a partial snapshot (e.g. a `.0.hdf5` file) or
        a full/virtual snapsoht covering all particles. This must be computed at run-time.
        """
        return False

    @staticmethod
    @abstractmethod
    def get_nice_name(group: str) -> str:
        """
        Converts the group name to a 'nice name' (i.e. for printing) for the SWIFTsimIO objects.
        """
        raise NotImplementedError


class MassTable(object):
    """
    Extracts a mass table to local variables based on the
    particle type names.
    """

    def __init__(self, base_mass_table: np.array, mass_units: unyt.unyt_quantity):
        """
        Parameters
        ----------

        base_mass_table : np.array
            Mass table of the same length as the number of particle types.

        mass_units : unyt_quantity
            Base mass units for the simulation.
        """

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

    def __str__(self):
        return f"Mass table for {' '.join(metadata.particle_types.particle_name_underscores.values())}"

    def __repr__(self):
        return self.__str__()


class MappingTable(object):
    """
    A mapping table from one named column instance to the other.
    Initially designed for the mapping between dust and elements.
    """

    def __init__(
        self,
        data: np.ndarray,
        named_columns_x: List[str],
        named_columns_y: List[str],
        named_columns_x_name: str,
        named_columns_y_name: str,
    ):
        """
        Parameters
        ----------

        data: np.ndarray
            The data array providing the mapping between the named
            columns. Should be of size N x M, where N is the number
            of elements in ``named_columns_x`` and M the number
            of elements in ``named_columns_y``.

        named_columns_x: List[str]
            The names of the columns in the first axis.

        named_columns_y: List[str]
            The names of the columns in the second axis.

        named_columns_x_name: str
            The name of the first mapping.

        named_columns_y_name: str
            The name of the second mapping.
        """

        self.data = data
        self.named_columns_x = named_columns_x
        self.named_columns_y = named_columns_y
        self.named_columns_x_name = named_columns_x_name
        self.named_columns_y_name = named_columns_y_name

        for x, name_x in enumerate(named_columns_x):
            for y, name_y in enumerate(named_columns_y):
                setattr(self, f"{name_x.lower()}_to_{name_y.lower()}", data[x][y])

        return

    def __str__(self):
        return (
            f"Mapping table from {self.named_columns_x_name} to "
            f"{self.named_columns_y_name}, containing {len(self.data)} "
            f"by {len(self.data[0])} elements."
        )

    def __repr__(self):
        return f"{self.__str__()}. Raw data: " "\n" f"{self.data}."


class SWIFTGroupMetadata(object):
    """
    Object that contains the metadata for one hdf5 group.

    This, for instance, could be part type 0, or 'gas'. This will load in
    the names of all datasets, their units, possible named fields,
    and their cosmology, and present them for use in the actual i/o routines.

    Methods
    -------
    load_metadata(self):
        Loads the required metadata.
    load_field_names(self):
        Loads in the field names.
    load_field_units(self):
        Loads in the units from each dataset.
    load_field_descriptions(self):
        Loads in descriptions of the fields for each dataset.
    load_field_compressions(self):
        Loads in compressions of the fields for each dataset.
    load_cosmology(self):
        Loads in the field cosmologies.
    load_physical(self):
        Loads in whether the field is saved as comoving or physical.
    load_valid_transforms(self):
        Loads in whether the field can be converted to comoving.
    load_named_columns(self):
        Loads the named column data for relevant fields.
    """

    def __init__(
        self,
        group: str,
        group_name: str,
        metadata: "SWIFTMetadata",
        scale_factor: float,
    ):
        """
        Constructor for SWIFTGroupMetadata class

        Parameters
        ----------
        group: str
            the name of the group in the hdf5 file
        group_name : str
            the corresponding group name for swiftsimio
        metadata : SWIFTMetadata
            the snapshot metadata
        scale_factor : float
            the snapshot scale factor
        """
        self.group = group
        self.group_name = group_name
        self.metadata = metadata
        self.units = metadata.units
        self.scale_factor = scale_factor

        self.filename = metadata.filename

        self.load_metadata()

        return

    def __str__(self):
        return f"Metadata class for {self.group} ({self.group_name})"

    def __repr__(self):
        return self.__str__()

    def load_metadata(self):
        """
        Loads the required metadata.

        This includes loading the field names, units and descriptions, as well as the
        cosmology metadata and any custom named columns
        """

        self.load_field_names()
        self.load_field_units()
        self.load_field_descriptions()
        self.load_field_compressions()
        self.load_cosmology()
        self.load_physical()
        self.load_valid_transforms()
        self.load_named_columns()

    def load_field_names(self):
        """
        Loads in only the field names.
        """

        # regular expression for camel case to snake case
        # https://stackoverflow.com/a/1176023
        def convert(name):
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        # Skip fields which are groups themselves
        self.field_paths = []
        self.field_names = []
        for item in self.metadata.handle[f"{self.group}"].keys():
            # Skip fields which are groups themselves
            if f"{self.group}/{item}" not in self.metadata.present_groups:
                self.field_paths.append(f"{self.group}/{item}")
                self.field_names.append(convert(item))

        return

    def load_field_units(self):
        """
        Loads in the units from each dataset.
        """

        unit_dict = {
            "I": self.units.current,
            "L": self.units.length,
            "M": self.units.mass,
            "T": self.units.temperature,
            "t": self.units.time,
        }

        def get_units(unit_attribute):
            units = 1.0

            for exponent, unit in unit_dict.items():
                # We store the 'unit exponent' in the SWIFT metadata. This corresponds
                # to the power we need to raise each unit to, to return the correct units
                try:
                    # Need to check if the exponent is 0 manually because of float precision
                    unit_exponent = unit_attribute[f"U_{exponent} exponent"][0]
                    if unit_exponent != 0.0:
                        units *= unit**unit_exponent
                except KeyError:
                    # Can't load that data!
                    # We should probably warn the user here...
                    pass

            # Deal with case where we _really_ have a dimensionless quantity. Comparing with
            # 1.0 doesn't work, beacause in these cases unyt reverts to a floating point
            # comparison.
            try:
                units.units
            except AttributeError:
                units = None

            return units

        self.field_units = [
            get_units(self.metadata.handle[x].attrs) for x in self.field_paths
        ]

        return

    def load_field_descriptions(self):
        """
        Loads in the text descriptions of the fields for each dataset.
        For SOAP filetypes a description of the mask is included.
        """

        def get_desc(dataset):
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
                mask_str = f" Only computed for objects with {mask_datasets[0]} >= {mask_threshold}."
            else:
                mask_str = f' Only computed for objects where {" + ".join(mask_datasets)} >= {mask_threshold}.'
            return description + mask_str

        self.field_descriptions = [
            get_desc(self.metadata.handle[x]) for x in self.field_paths
        ]

        return

    def load_field_compressions(self):
        """
        Loads in the string describing the compression filters of the fields for each dataset.
        """

        def get_comp(dataset):
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

        self.field_compressions = [
            get_comp(self.metadata.handle[x]) for x in self.field_paths
        ]

        return

    def load_cosmology(self):
        """
        Loads in the field cosmologies.
        """

        current_scale_factor = self.scale_factor

        def get_cosmo(dataset):
            try:
                cosmo_exponent = dataset.attrs["a-scale exponent"][0]
            except:
                # Can't load, 'graceful' fallback.
                cosmo_exponent = 0.0

            a_factor_this_dataset = a**cosmo_exponent

            return cosmo_factor(a_factor_this_dataset, current_scale_factor)

        self.field_cosmologies = [
            get_cosmo(self.metadata.handle[x]) for x in self.field_paths
        ]

        return

    def load_physical(self):
        """
        Loads in whether the field is saved as comoving or physical.
        """

        def get_physical(dataset):
            try:
                physical = dataset.attrs["Value stored as physical"][0] == 1
            except:
                physical = False
            return physical

        self.field_physicals = [
            get_physical(self.metadata.handle[x]) for x in self.field_paths
        ]

        return

    def load_valid_transforms(self):
        """
        Loads in whether the field can be converted to comoving.
        """

        def get_valid_transform(dataset):
            try:
                valid_transform = (
                    dataset.attrs["Property can be converted to comoving"][0] == 1
                )
            except:
                valid_transform = True
            return valid_transform

        self.field_valid_transforms = [
            get_valid_transform(self.metadata.handle[x]) for x in self.field_paths
        ]

        return

    def load_named_columns(self):
        """
        Loads the named column data for relevant fields.
        """

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

                num_capitals = lambda x: sum(1 for c in x if c.isupper())
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


class SWIFTUnits(object):
    """
    Generates a unyt system that can then be used with the SWIFT data.

    These give the unit mass, length, time, current, and temperature as
    unyt unit variables in simulation units. I.e. you can take any value
    that you get out of the code and multiply it by the appropriate values
    to get it 'unyt-ified' with the correct units.

    Attributes
    ----------
    mass : float
        unit for mass used
    length : float
        unit for length used
    time : float
        unit for time used
    current : float
        unit for current used
    temperature : float
        unit for temperature used

    """

    def __init__(self, filename: Path, handle: Optional[h5py.File] = None):
        """
        SWIFTUnits constructor

        Sets filename for file to read units from and gets unit dictionary

        Parameters
        ----------

        filename : Path
            Name of file to read units from

        handle: h5py.File, optional
            The h5py file handle, optional. Will open a new handle with the
            filename if required.

        """
        self.filename = filename
        self._handle = handle

        self.get_unit_dictionary()

        return

    @property
    def handle(self):
        """
        Property that gets the file handle, which can be shared
        with other objects for efficiency reasons.
        """
        if isinstance(self._handle, h5py.File):
            # Can be open or closed, let's test.
            try:
                file = self._handle.file

                return self._handle
            except ValueError:
                # This will be the case if there is no active file handle
                pass

        self._handle = h5py.File(self.filename, "r")

        return self._handle

    def get_unit_dictionary(self):
        """
        Store unit data and metadata

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

    def __del__(self):
        if isinstance(self._handle, h5py.File):
            self._handle.close()


def metadata_discriminator(filename: str, units: SWIFTUnits) -> "SWIFTMetadata":
    """
    Discriminates between the different types of metadata objects read from SWIFT-compatile
    files.

    Parameters
    ----------

    filename : str
        Name of the file to read metadata from

    units : SWIFTUnits
        The units object associated with the file


    Returns
    -------

    SWIFTMetadata
        The appropriate metadata object for the file type
    """
    # Old snapshots did not have this attribute, so we need to default to FullVolume
    file_type = units.handle["Header"].attrs.get("OutputType", "FullVolume")

    if isinstance(file_type, bytes):
        file_type = file_type.decode("utf-8")

    if file_type in ["FullVolume"]:
        return SWIFTSnapshotMetadata(filename, units)
    elif file_type in ["SOAP"]:
        return SWIFTSOAPMetadata(filename, units)
    elif file_type in ["FOF"]:
        return SWIFTFOFMetadata(filename, units)
    else:
        raise ValueError(f"File type {file_type} not recognised.")


class SWIFTSnapshotMetadata(SWIFTMetadata):
    """
    SWIFT Metadata for a snapshot-style file containing particle
    information. For more documentation, see the main :cls:`SWIFTMetadata`
    class.
    """

    masking_valid: bool = True

    def __init__(self, filename, units: SWIFTUnits):
        """
        Constructor for SWIFTMetadata object

        Parameters
        ----------

        filename : str
            name of file to read from

        units : SWIFTUnits
            the units being used
        """
        self.filename = filename
        self.units = units

        self.get_metadata()
        self.get_named_column_metadata()
        self.get_mapping_metadata()

        self.postprocess_header()

        self.load_groups()
        self.extract_cosmology()

        # After we've loaded all this metadata, we can safely release the file handle.
        self.handle.close()

        return

    def get_named_column_metadata(self):
        """
        Loads the custom named column metadata (if it exists) from
        SubgridScheme/NamedColumns.
        """

        try:
            data = self.handle["SubgridScheme/NamedColumns"]

            self.named_columns = {
                k: [x.decode("utf-8") for x in data[k][:]] for k in data.keys()
            }
        except KeyError:
            self.named_columns = {}

        return

    def get_mapping_metadata(self):
        """
        Gets the mappings based on the named columns (must have already been read),
        from the form:

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

        # regular expression for camel case to snake case
        # https://stackoverflow.com/a/1176023
        def convert(name):
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        regex = r"([a-zA-Z]*)To([a-zA-Z]*)Mapping"
        compiled = re.compile(regex)

        for key, data in zip(available_keys, available_data):
            match = compiled.match(key)
            snake_case = convert(key)

            if match:
                x = match.group(1)
                y = match.group(2)

                if x == "Grain":
                    warnings.warn(
                        "Use of the GrainToElementMapping is deprecated, please use a newer "
                        "version of SWIFT to run this simulation.",
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
    def present_groups(self):
        """
        The groups containing datasets that are present in the file.
        """
        types = np.where(np.array(getattr(self, "has_type", self.num_part)) != 0)[0]
        return [f"PartType{i}" for i in types]

    @property
    def present_group_names(self):
        """
        The names of the groups that we want to expose.
        """

        return [
            metadata.particle_types.particle_name_underscores[x]
            for x in self.present_groups
        ]

    @property
    def code_info(self) -> str:
        """
        Gets a nicely printed set of code information with:

        Name (Git Branch)
        Git Revision
        Git Date
        """

        def get_string(x):
            return self.code[x].decode("utf-8")

        output = (
            f"{get_string('Code')} ({get_string('Git Branch')})\n"
            f"{get_string('Git Revision')}\n"
            f"{get_string('Git Date')}"
        )

        return output

    @property
    def compiler_info(self) -> str:
        """
        Gets information about the compiler and formats it as:

        Compiler Name (Compiler Version)
        MPI library
        """

        def get_string(x):
            return self.code[x].decode("utf-8")

        output = (
            f"{get_string('Compiler Name')} ({get_string('Compiler Version')})\n"
            f"{get_string('MPI library')}"
        )

        return output

    @property
    def library_info(self) -> str:
        """
        Gets information about the libraries used and formats it as:

        FFTW vFFTW library version
        GSL vGSL library version
        HDF5 vHDF5 library version
        """

        def get_string(x):
            return self.code[f"{x} library version"].decode("utf-8")

        output = (
            f"FFTW v{get_string('FFTW')}\n"
            f"GSL v{get_string('GSL')}\n"
            f"HDF5 v{get_string('HDF5')}"
        )

        return output

    @property
    def hydro_info(self) -> str:
        r"""
        Gets information about the hydro scheme and formats it as:

        Scheme
        Kernel function in DimensionD
        $\eta$ = Kernel eta (Kernel target N_ngb $N_{ngb}$)
        $C_{\rm CFL}$ = CFL parameter
        """

        def get_float(x):
            return "{:4.2f}".format(self.hydro_scheme[x][0])

        def get_int(x):
            return int(self.hydro_scheme[x][0])

        def get_string(x):
            return self.hydro_scheme[x].decode("utf-8")

        output = (
            f"{get_string('Scheme')}\n"
            f"{get_string('Kernel function')} in {get_int('Dimension')}D\n"
            rf"$\eta$ = {get_float('Kernel eta')} "
            rf"({get_float('Kernel target N_ngb')} $N_{{ngb}}$)"
            "\n"
            rf"$C_{{\rm CFL}}$ = {get_float('CFL parameter')}"
        )

        return output

    @property
    def viscosity_info(self) -> str:
        r"""
        Gets information about the viscosity scheme and formats it as:

        Viscosity Model
        $\alpha_{V, 0}$ = Alpha viscosity, $\ell_V$ = Viscosity decay length [internal units], $\beta_V$ = Beta viscosity
        Alpha viscosity (min) < $\alpha_V$ < Alpha viscosity (max)
        """

        def get_float(x):
            return "{:4.2f}".format(self.hydro_scheme[x][0])

        def get_string(x):
            return self.hydro_scheme[x].decode("utf-8")

        output = (
            f"{get_string('Viscosity Model')}\n"
            rf"$\alpha_{{V, 0}}$ = {get_float('Alpha viscosity')}, "
            rf"$\ell_V$ = {get_float('Viscosity decay length [internal units]')}, "
            rf"$\beta_V$ = {get_float('Beta viscosity')}"
            "\n"
            rf"{get_float('Alpha viscosity (min)')} < $\alpha_V$ < {get_float('Alpha viscosity (max)')}"
        )

        return output

    @property
    def diffusion_info(self) -> str:
        """
        Gets information about the diffusion scheme and formats it as:

        $\alpha_{D, 0}$ = Diffusion alpha, $\beta_D$ = Diffusion beta
        Diffusion alpha (min) < $\alpha_D$ < Diffusion alpha (max)
        """

        def get_float(x):
            return "{:4.2f}".format(self.hydro_scheme[x][0])

        output = (
            rf"$\alpha_{{D, 0}}$ = {get_float('Diffusion alpha')}, "
            rf"$\beta_D$ = {get_float('Diffusion beta')}"
            "\n"
            rf"${get_float('Diffusion alpha (min)')} < "
            rf"\alpha_D < {get_float('Diffusion alpha (max)')}$"
        )

        return output

    @property
    def partial_snapshot(self) -> bool:
        """
        Whether or not this snapshot is partial (e.g. a "x.0.hdf5" file), or
        a file describing an entire snapshot.
        """

        # Partial snapshots have num_files_per_snapshot set to 1. Virtual snapshots
        # collating multiple sub-snapshots together have num_files_per_snapshot = 1.

        return self.num_files_per_snapshot > 1

    @staticmethod
    def get_nice_name(group):
        return metadata.particle_types.particle_name_class[group]


class SWIFTFOFMetadata(SWIFTMetadata):
    """
    SWIFT Metadata for a snapshot-style file containing particle
    information. For more documentation, see the main :cls:`SWIFTMetadata`
    class.
    """

    homogeneous_arrays: bool = True

    def __init__(self, filename: str, units: SWIFTUnits):
        self.filename = filename
        self.units = units

        self.get_metadata()
        self.postprocess_header()

        self.load_groups()

        # After we've loaded all this metadata, we can safely release the file handle.
        self.handle.close()

        return

    @property
    def present_groups(self):
        """
        The groups containing datasets that are present in the file.
        """
        return ["Groups"]

    @property
    def present_group_names(self):
        """
        The names of the groups that we want to expose.
        """
        return ["fof_groups"]

    @staticmethod
    def get_nice_name(group):
        return "FOFGroups"


class SWIFTSOAPMetadata(SWIFTMetadata):
    """
    SWIFT Metadata for a snapshot-style file containing particle
    information. For more documentation, see the main :cls:`SWIFTMetadata`
    class.
    """

    masking_valid: bool = True
    shared_cell_counts: str = "Subhalos"
    homogeneous_arrays: bool = True

    def __init__(self, filename: str, units: SWIFTUnits):
        self.filename = filename
        self.units = units

        self.get_metadata()
        self.postprocess_header()
        self.unpack_subhalo_number()

        self.load_groups()

        # After we've loaded all this metadata, we can safely release the file handle.
        self.handle.close()

        return

    def unpack_subhalo_number(self):
        self.n_subhalos = int(self.num_subhalo[0])

    @property
    def present_groups(self):
        """
        The groups containing datasets that are present in the file.
        """
        return self.subhalo_types

    @property
    def present_group_names(self):
        """
        The names of the groups that we want to expose.
        """
        return [
            metadata.soap_types.get_soap_name_underscore(x) for x in self.present_groups
        ]

    @staticmethod
    def get_nice_name(group):
        return metadata.soap_types.get_soap_name_nice(group)
