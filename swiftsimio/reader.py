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

from typing import Union, Callable, List


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

    def __init__(self, filename):
        """
        SWIFTUnits constructor

        Sets filename for file to read units from and gets unit dictionary

        Parameters
        ----------
        filename : str
            name of file to read units from
        """
        self.filename = filename

        self.get_unit_dictionary()

        return

    def get_unit_dictionary(self):
        """
        Store unit data and metadata

        Length 1 arrays are used to store the unit data. This dictionary 
        also contains the metadata information that connects the unyt 
        objects to the names that are stored in the SWIFT snapshots.
        """
        with h5py.File(self.filename, "r") as handle:
            self.units = {
                name: value[0] * metadata.unit_types.unit_names_to_unyt[name]
                for name, value in handle["Units"].attrs.items()
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


class SWIFTMetadata(object):
    """
    Loads all metadata (apart from Units, those are handled by SWIFTUnits)
    into dictionaries. 
    
    This also does some extra parsing on some well-used metadata.

    Attributes
    ----------
    filename: str
        name of file to read from
    units: SWIFTUnits
        the units being used
    header: dict
        stores metadata about snapshot
    present_particle_names:
        Get particle names present in snapshot
    code_info:
        Print info about SWIFT version used
    compiler_info:
        Print info about compilers used
    library_info:
        Print info about library versions used
    hydro_info:
        Print info about hydro scheme used
    viscosity_info
        Print info about viscosity scheme used
    diffusion_info:
        Print info about diffusion scheme used

    Methods
    -------
    get_metadata(self):
        Loads the metadata 
    get_named_column_metadata(self):
        Loads custom metadata from named columns
    postprocess_header(self):
        Postprocesses local variables in header 
    present_particle_types(self):
        Get particle types present in snapshot
    """

    filename: str
    units: SWIFTUnits
    header: dict

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

    def get_named_column_metadata(self):
        """
        Loads the custom named column metadata (if it exists) from
        SubgridScheme/NamedColumns.
        """

        try:
            with h5py.File(self.filename, "r") as handle:
                data = handle["SubgridScheme/NamedColumns"]

                self.named_columns = {
                    k: [x.decode("utf-8") for x in data[k][:]] for k in data.keys()
                }
        except KeyError:
            self.named_columns = {}

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
                    # This is required or we automatically get everything in CGS!
                    getattr(self, name).convert_to_units(
                        header_unpack_variables_units[name]
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
            try:
                self.snapshot_date = datetime.strptime(
                    self.header["Snapshot date"].decode("utf-8"), "%H:%M:%S %Y-%m-%d %Z"
                )
            except ValueError:
                # Backwards compatibility; this was used previously due to simplicity
                # but is not portable between regions. So if you ran a simulation on
                # a British (en_GB) machine, and then tried to read on a Dutch
                # machine (nl_NL), this would _not_ work because %c is different.
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
    def viscosity_info(self):
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
    def diffusion_info(self):
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


class SWIFTParticleTypeMetadata(object):
    """
    Object that contains the metadata for one particle type. 
    
    This, for instance, could be part type 0, or 'gas'. This will load in 
    the names of all particle datasets, their units, possible named fields, 
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
    load_cosmology(self):
        Loads in the field cosmologies.
    load_named_columns(self):
        Loads the named column data for relevant fields.
    """

    def __init__(
        self,
        particle_type: int,
        particle_name: str,
        metadata: SWIFTMetadata,
        scale_factor: float,
    ):
        """
        Constructor for SWIFTParticleTypeMetadata class

        Parameters
        ----------
        partycle_type : int
            the integer particle type
        particle_name : str
            the corresponding particle name
        metadata : SWIFTMetadata
            the snapshot metadata
        scale_factor : float
            the snapshot scale factor
        """
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
        self.load_cosmology()
        self.load_named_columns()

    def load_field_names(self):
        """
        Loads in only the field names.
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

    def load_named_columns(self):
        """
        Loads the named column data for relevant fields.
        """

        named_columns = {}

        for field in self.field_paths:
            property_name = field.split("/")[-1]

            if property_name in self.metadata.named_columns.keys():
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


def generate_getter(
    filename,
    name: str,
    field: str,
    unit: unyt.unyt_quantity,
    mask: Union[None, np.ndarray],
    mask_size: int,
    cosmo_factor: cosmo_factor,
    description: str,
    columns: np.lib.index_tricks.IndexExpression = np.s_[:],
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

    columns: np.lib.index_tricks.IndexEpression, optional
        Index expression corresponding to which columns to read from the numpy array.
        If not provided, we read all columns and return an n-dimensional array.

    
    Returns
    -------

    getter: callable
        A callable object that gets the value of the array that has been saved to
        ``_name``. This function takes only ``self`` from the 
        :obj:``__SWIFTParticleDataset`` class.


    Notes
    -----

    The major use of this function is for its side effect of setting ``_name`` as
    a member of the class on first read. When the attribute is accessed, it will
    be dynamically read from the file, to keep initial memory usage as minimal
    as possible.

    If the resultant array is modified, it will not be re-read from the file.

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
                                    columns=columns,
                                ),
                                unit,
                                cosmo_factor=cosmo_factor,
                                name=description,
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


class __SWIFTParticleDataset(object):
    """
    Creates empty property fields

    Do not use this class alone; it is essentially completely empty. It is filled
    with properties by generate_dataset.

    Methods
    -------
    generate_empty_properties(self)
        creates empty properties to be accessed through setter and getter functions
    """

    def __init__(self, particle_metadata: SWIFTParticleTypeMetadata):
        """
        Constructor for SWIFTParticleDataset class

        This function primarily calls the generate_empty_properties
        function to ensure that defaults are set correctly.

        Parameters
        ----------
        particle_metadata : SWIFTParticleTypeMetadata
            the metadata used to generate empty properties
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
        setter and getters. 
        
        Initially set all of the _{name} values to None. If it doesn't 
        _exist_ in the file, the variable is not created.
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


class __SWIFTNamedColumnDataset(object):
    """
    Holder class for individual named datasets. Very similar to
    __SWIFTParticleDataset but much simpler.
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

    Parameters
    ----------
    particle_metadata : SWIFTParticleTypeMetadata
        the metadata for the particle type
    mask : SWIFTMask
        the mask object for the dataset
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
    field_named_columns = particle_metadata.named_columns

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
                ),
                generate_setter(field_name),
                generate_deleter(field_name),
            )
        else:
            # TODO: Handle this case with recursion.

            # Here we want to create an extra middleman object. So we can do something
            # like {ptype}.{ThisNamedColumnDataset}.column_name. This follows from the
            # above templating.
            ThisNamedColumnDataset = type(
                f"{particle_nice_name}{field_path.split('/')[-1]}Columns",
                __SWIFTNamedColumnDataset.__bases__,
                dict(__SWIFTNamedColumnDataset.__dict__),
            )

            for index, column in enumerate(named_columns):
                setattr(
                    ThisNamedColumnDataset,
                    column,
                    property(
                        generate_getter(
                            filename,
                            column,
                            field_path,
                            unit=field_unit,
                            mask=mask_array,
                            mask_size=mask_size,
                            cosmo_factor=field_cosmology,
                            description=f"{field_description} [Column {index}, {column}]",
                            columns=np.s_[index],
                        ),
                        generate_setter(column),
                        generate_deleter(column),
                    ),
                )

            field_property = ThisNamedColumnDataset(
                field_path=field_path,
                named_columns=named_columns,
                name=field_description,
            )

        setattr(ThisDataset, field_name, field_property)

    empty_dataset = ThisDataset(particle_metadata)

    return empty_dataset


class SWIFTDataset(object):
    """
    A collection object for:

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

    Methods
    -------
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
        self.create_particle_datasets()

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
