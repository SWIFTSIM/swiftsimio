"""
This file contains four major objects:

+ SWIFTUnits, which is a unit system that can be queried for units (and converts arrays
  to relevant unyt arrays when read from the HDF5 file)
+ SWIFTMetadata, which contains all of the metadata from the file
+ __SWIFTParticleDataset, which contains particle information but should never be
  directly accessed. Use generate_dataset to create one of these. The reasoning
  here is that properties can only be added to the class afterwards, and not
  directly in an _instance_ of the class.
+ SWIFTDataset, a container class for all of the above.
"""

from swiftsimio import metadata
from swiftsimio.accelerated import read_ranges_from_file
from swiftsimio.objects import cosmo_array, cosmo_factor, a

import re
import h5py
import unyt
import numpy as np


from datetime import datetime

from typing import Union, Callable


class SWIFTUnits(object):
    """
    Generates a unyt system that can then be used with the SWIFT data.

    You are probably looking for the following attributes:

    + SWIFTUnits.mass
    + SWIFTUnits.length
    + SWIFTUnits.time
    + SWIFTUnits.current
    + SWIFTUnits.temperature

    That give the unit mass, length, time, current, and temperature as
    unyt unit variables in simulation units. I.e. you can take any value
    that you get out of the code and multiply it by the appropriate values
    to get it 'unyt-ified' with the correct units.
    """

    def __init__(self, filename):
        self.filename = filename

        self.get_unit_dictionary()

        return

    def get_unit_dictionary(self):
        """
        For some reason instead of just floats we use length 1 arrays to
        store the unit data. This dictionary also happens to contain the
        metadata information that connects the unyt objects to the names
        that are stored in the SWIFT snapshots.
        """
        with h5py.File(self.filename, "r") as handle:
            self.units = {
                name: value[0] * metadata.unit_types.unit_names_to_unyt[name]
                for name, value in handle["Units"].attrs.items()
            }

        # We now unpack this into variables.
        self.mass = self.units["Unit mass in cgs (U_M)"]
        self.length = self.units["Unit length in cgs (U_L)"]
        self.time = self.units["Unit time in cgs (U_t)"]
        self.current = self.units["Unit current in cgs (U_I)"]
        self.temperature = self.units["Unit temperature in cgs (U_T)"]


class SWIFTMetadata(object):
    """
    Loads all metadata (apart from Units, those are handled by SWIFTUnits)
    into dictionaries. This also does some extra parsing on some well-used
    metadata.
    """

    def __init__(self, filename, units: SWIFTUnits):
        self.filename = filename
        self.units = units

        self.get_metadata()

        self.postprocess_header()

        self.load_particle_types()

        return

    def get_metadata(self):
        """
        Loads the metadata as specified in metadata.metadata_fields.
        """

        with h5py.File(self.filename, "r") as handle:
            for field, name in metadata.metadata_fields.metadata_fields_to_read.items():
                try:
                    setattr(self, name, dict(handle[field].attrs))
                except KeyError:
                    setattr(self, name, None)

        return

    def postprocess_header(self):
        """
        Some minor postprocessing on the header to local variables.
        """

        # These are just read straight in to variables
        header_unpack_variables_units = metadata.metadata_fields.generate_units_header_unpack_variables(
            m=self.units.mass,
            l=self.units.length,
            t=self.units.time,
            I=self.units.current,
            T=self.units.temperature,
        )

        for field, name in metadata.metadata_fields.header_unpack_variables.items():
            try:
                if name in header_unpack_variables_units.keys():
                    setattr(
                        self,
                        name,
                        self.header[field] * header_unpack_variables_units[name],
                    )
                else:
                    # Must not have any units! Oh well.
                    setattr(self, name, self.header[field])
            except KeyError:
                # Must not be present, just skip it
                continue

        # These must be unpacked as 'real' strings (i.e. converted to utf-8)

        for field, name in metadata.metadata_fields.header_unpack_string.items():
            try:
                setattr(self, name, self.header[field])
            except KeyError:
                # Must not be present, just skip it
                continue

        # These must be unpacked as they are stored as length-1 arrays

        header_unpack_float_units = metadata.metadata_fields.generate_units_header_unpack_single_float(
            m=self.units.mass,
            l=self.units.length,
            t=self.units.time,
            I=self.units.current,
            T=self.units.temperature,
        )

        for field, names in metadata.metadata_fields.header_unpack_single_float.items():
            try:
                if isinstance(names, list):
                    # Sometimes we store a list in case we have multiple names, for example
                    # Redshift -> metadata.redshift AND metadata.z. Can't just do the iteration
                    # because we may loop over the letters in the string.
                    for variable in names:
                        if variable in header_unpack_float_units.keys():
                            # We have an associated unit!
                            unit = header_unpack_float_units[variable]
                            setattr(self, variable, self.header[field][0] * unit)
                        else:
                            # No unit
                            setattr(self, variable, self.header[field][0])
                else:
                    # We can just check for the unit and set the attribute
                    variable = names
                    if variable in header_unpack_float_units.keys():
                        # We have an associated unit!
                        unit = header_unpack_float_units[variable]
                        setattr(self, variable, self.header[field][0] * unit)
                    else:
                        # No unit
                        setattr(self, variable, self.header[field][0])
            except KeyError:
                # Must not be present, just skip it
                continue

        # These are special cases, sorry!
        # Date and time of snapshot dump
        try:
            self.snapshot_date = datetime.strptime(
                self.header["Snapshot date"].decode("utf-8"), "%c\n"
            )
        except KeyError:
            # Old file
            pass

        # Store these separately as self.n_gas = number of gas particles for example
        for (
            part_number,
            part_name,
        ) in metadata.particle_types.particle_name_underscores.items():
            setattr(self, f"n_{part_name}", self.num_part[part_number])

        # Need to unpack the gas gamma for cosmology
        try:
            self.gas_gamma = self.hydro_scheme["Adiabatic index"]
        except (KeyError, TypeError):
            print("Could not find gas gamma, assuming 5./3.")
            self.gas_gamma = 5.0 / 3.0

        try:
            self.a = self.scale_factor
        except AttributeError:
            # These must always be present for the initialisation of cosmology properties
            self.a = 1.0
            self.scale_factor = 1.0

        return

    def load_particle_types(self):
        """
        Loads the particle types and metadata into objects:

            metadata.<type>_properties

        This contains five arrays,

            metadata.<type>_properties.field_names
            metadata.<type>_properties.field_paths
            metadata.<type>_properties.field_units
            metadata.<type>_properties.field_cosmologies
            metadata.<type>_properties.field_descriptions

        As well as some more information about the particle type.
        """

        for particle_type, particle_name in zip(
            self.present_particle_types, self.present_particle_names
        ):
            setattr(
                self,
                f"{particle_name}_properties",
                SWIFTParticleTypeMetadata(
                    particle_type=particle_type,
                    particle_name=particle_name,
                    metadata=self,
                    scale_factor=self.scale_factor,
                ),
            )

        return

    @property
    def present_particle_types(self):
        """
        The particle types that are present in the file.
        """

        return np.where(np.array(self.num_part) != 0)[0]

    @property
    def present_particle_names(self):
        """
        The particle _names_ that are present in the simulation.
        """

        return [
            metadata.particle_types.particle_name_underscores[x]
            for x in self.present_particle_types
        ]

    @property
    def code_info(self):
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
    def compiler_info(self):
        """
        Gets information about the compiler and formats it in a string like:

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
    def library_info(self):
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
    def hydro_info(self):
        r"""
        Grabs information about the hydro scheme and formats it nicely:

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
    def viscosity_info(self):
        r"""
        Pretty-prints some information about the viscosity scheme.

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
    def diffusion_info(self):
        """
        Pretty-prints some information about the diffusion scheme.

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


class SWIFTParticleTypeMetadata(object):
    """
    Object that contains the metadata for one particle type. This, for, instance,
    could be part type 0, or 'gas'. This will load in the names of all particle datasets,
    their units, and their cosmology, and present them for use in the actual
    i/o routines.
    """

    def __init__(
        self,
        particle_type: int,
        particle_name: str,
        metadata: SWIFTMetadata,
        scale_factor: float,
    ):
        self.particle_type = particle_type
        self.particle_name = particle_name
        self.metadata = metadata
        self.units = metadata.units
        self.scale_factor = scale_factor

        self.filename = metadata.filename

        self.load_metadata()

        return

    def __str__(self):
        return f"Metadata class for PartType{self.particle_type} ({self.particle_name})"

    def load_metadata(self):
        """
        Workhorse function, loads the requried metadata.
        """

        self.load_field_names()
        self.load_field_units()
        self.load_field_descriptions()
        self.load_cosmology()

    def load_field_names(self):
        """
        Loads in only the field names (including dealing with recursion).
        """

        # regular expression for camel case to snake case
        # https://stackoverflow.com/a/1176023
        def convert(name):
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        with h5py.File(self.filename, "r") as handle:
            self.field_paths = [
                f"PartType{self.particle_type}/{item}"
                for item in handle[f"PartType{self.particle_type}"].keys()
            ]

            self.field_names = [
                convert(item) for item in handle[f"PartType{self.particle_type}"].keys()
            ]

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
                        units *= unit ** unit_exponent
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

        with h5py.File(self.filename, "r") as handle:
            self.field_units = [get_units(handle[x].attrs) for x in self.field_paths]

        return

    def load_field_descriptions(self):
        """
        Loads in the text descriptions of the fields for each dataset.
        """

        def get_desc(dataset):
            try:
                description = dataset.attrs["Description"].decode("utf-8")
            except KeyError:
                # Can't load description!
                description = "No description available"

            return description

        with h5py.File(self.filename, "r") as handle:
            self.field_descriptions = [get_desc(handle[x]) for x in self.field_paths]

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

            a_factor_this_dataset = a ** cosmo_exponent

            return cosmo_factor(a_factor_this_dataset, current_scale_factor)

        with h5py.File(self.filename, "r") as handle:
            self.field_cosmologies = [get_cosmo(handle[x]) for x in self.field_paths]

        return


def generate_getter(
    filename,
    name: str,
    field: str,
    unit,
    mask: Union[None, np.ndarray],
    mask_size: int,
    cosmo_factor: cosmo_factor,
    description: str,
):
    """
    Generates a function that:

    a) If self._`name` exists, return it
    b) If not, open `filename`
    c) Reads filename[`field`]
    d) Set self._`name`
    e) Return self._`name`.
    """

    def getter(self):
        current_value = getattr(self, f"_{name}")

        if current_value is not None:
            return current_value
        else:
            with h5py.File(filename, "r") as handle:
                try:
                    if mask is not None:
                        # First, need to claculate data shape (which may be
                        # non-trivial), so we read in the first value
                        first_value = handle[field][0]

                        output_type = first_value.dtype
                        output_size = first_value.size

                        if output_size != 1:
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
                                ),
                                unit,
                                cosmo_factor=cosmo_factor,
                                description=description,
                            ),
                        )
                    else:
                        setattr(
                            self,
                            f"_{name}",
                            cosmo_array(
                                handle[field][...],
                                unit,
                                cosmo_factor=cosmo_factor,
                                description=description,
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
    """

    def setter(self, value):
        setattr(self, f"_{name}", value)

        return

    return setter


def generate_deleter(name: str):
    """
    Generates a function that destroys self._name (sets it back to None).
    """

    def deleter(self):
        current_value = getattr(self, f"_{name}")
        del current_value
        setattr(self, f"_{name}", None)

        return

    return deleter


class __SWIFTParticleDataset(object):
    """
    Do not use this class alone; it is essentially completely empty. It is filled
    with properties by generate_dataset.
    """

    def __init__(self, particle_metadata: SWIFTParticleTypeMetadata):
        """
        This function primarily calls the generate_getters_for_particle_types
        function to ensure that we are reading correctly.
        """
        self.filename = particle_metadata.filename
        self.units = particle_metadata.units

        self.particle_type = particle_metadata.particle_type
        self.particle_name = particle_metadata.particle_name

        self.particle_metadata = particle_metadata
        self.metadata = particle_metadata.metadata

        self.generate_empty_properties()

        return

    def generate_empty_properties(self):
        """
        Generates the empty properties that will be accessed through the
        setter and getters. We initially set all of the _{name} values
        to None. If it doesn't _exist_ in the file, we do not create the
        variable.
        """

        with h5py.File(self.filename, "r") as handle:
            for field_name, field_path in zip(
                self.particle_metadata.field_names, self.particle_metadata.field_paths
            ):
                if field_path in handle:
                    setattr(self, f"_{field_name}", None)
                else:
                    raise AttributeError(
                        (
                            f"Cannot find attribute {field_path} in file although"
                            "it was present when searching initially."
                        )
                    )

        return


def generate_dataset(particle_metadata: SWIFTParticleTypeMetadata, mask):
    """
    Generates a SWIFTParticleDataset _class_ that corresponds to the
    particle type given.

    We _must_ do the following _outside_ of the class itself, as one
    can assign properties to a _class_ but not _within_ a class
    dynamically.

    Here we loop through all of the possible properties in the metadata file.
    We then use the builtin property() function and some generators to
    create setters and getters for those properties. This will allow them
    to be accessed from outside by using SWIFTParticleDataset.name, where
    the name is, for example, coordinates.
    """

    filename = particle_metadata.filename
    particle_type = particle_metadata.particle_type
    particle_name = particle_metadata.particle_name
    particle_nice_name = metadata.particle_types.particle_name_class[particle_type]

    # Mask is an object that contains all masks for all possible datasets.
    if mask is not None:
        mask_array = getattr(mask, particle_name)
        mask_size = getattr(mask, f"{particle_name}_size")
    else:
        mask_array = None
        mask_size = -1

    # Set up an iterator for us to loop over for all fields
    field_paths = particle_metadata.field_paths
    field_names = particle_metadata.field_names
    field_cosmologies = particle_metadata.field_cosmologies
    field_units = particle_metadata.field_units
    field_descriptions = particle_metadata.field_descriptions

    dataset_iterator = zip(
        field_paths, field_names, field_cosmologies, field_units, field_descriptions
    )

    # This 'nice' piece of code ensures that our datasets have different _types_
    # for different particle types.
    ThisDataset = type(
        f"{particle_nice_name}Dataset",
        __SWIFTParticleDataset.__bases__,
        dict(__SWIFTParticleDataset.__dict__),
    )

    for (
        field_path,
        field_name,
        field_cosmology,
        field_unit,
        field_description,
    ) in dataset_iterator:
        setattr(
            ThisDataset,
            field_name,
            property(
                generate_getter(
                    filename,
                    field_name,
                    field_path,
                    unit=field_unit,
                    mask=mask_array,
                    mask_size=mask_size,
                    cosmo_factor=field_cosmology,
                    description=field_description,
                ),
                generate_setter(field_name),
                generate_deleter(field_name),
            ),
        )

    empty_dataset = ThisDataset(particle_metadata)

    return empty_dataset


class SWIFTDataset(object):
    """
    A collection object for the above three:

    + SWIFTUnits,
    + SWIFTMetadata,
    + SWIFTParticleDataset
    
    This object, in essence, completely represents a SWIFT snapshot. You can access
    the different particles as follows:

    + SWIFTDataset.gas.particle_ids
    + SWIFTDataset.dark_matter.masses
    + SWIFTDataset.gas.smoothing_lengths
    
    These arrays all have units that are determined by the unit system in the file.

    The unit system is available as SWIFTDataset.units and the metadata as
    SWIFTDataset.metadata.
    """

    def __init__(self, filename, mask=None):
        """
        Takes two arguments, the filename of the SWIFT dataset, including
        the file extension, and an optional mask object (of type SWIFTMask)
        should you wish to constrain the dataset to a given set of particles.
        """
        self.filename = filename
        self.mask = mask

        if mask is not None:
            self.mask.convert_masks_to_ranges()

        self.get_units()
        self.get_metadata()
        self.create_particle_datasets()

        return

    def __str__(self):
        """
        Prints out some more useful information, rather than just
        the memory location.
        """

        return f"SWIFT dataset at {self.filename}."

    def get_units(self):
        """
        Loads the units from the SWIFT snapshot. This happens automatically,
        but you can call this function again if you mess things up.
        """

        self.units = SWIFTUnits(self.filename)

        return

    def get_metadata(self):
        """
        Loads the metadata from the SWIFT snapshot. This happens automatically,
        but you can call this function again if you mess things up.
        """

        self.metadata = SWIFTMetadata(self.filename, self.units)

        return

    def create_particle_datasets(self):
        """
        Creates particle datasets for whatever particle types and names
        are specified in metadata.particle_types. These can then be
        accessed using their underscore names, e.g. gas.
        """

        if not hasattr(self, "metadata"):
            self.get_metadata()

        for particle_name in self.metadata.present_particle_names:
            setattr(
                self,
                particle_name,
                generate_dataset(
                    getattr(self.metadata, f"{particle_name}_properties"), self.mask
                ),
            )

        return
