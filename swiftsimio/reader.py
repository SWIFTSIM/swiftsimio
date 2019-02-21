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

import h5py
import unyt
import numpy as np

from datetime import datetime

from typing import Union


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
        self.generate_units()

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

    def generate_units(self):
        """
        Creates the unyt system to use to reduce data.
        """

        unit_fields = metadata.unit_fields.generate_units(
            m=self.mass,
            l=self.length,
            t=self.time,
            I=self.current,
            T=self.temperature,
        )

        for ptype, units in unit_fields.items():
            setattr(self, ptype, units)

        return


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
                    setattr(self, name, self.header[field] * header_unpack_variables_units[name])
                else:
                    # Must not have any units! Oh well.
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

        for field, name in metadata.metadata_fields.header_unpack_single_float.items():
            try:
                if name in header_unpack_float_units.keys():
                    setattr(self, name, self.header[field][0] * header_unpack_float_units[name])
                else:
                    setattr(self, name, self.header[field][0])
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

        # Store these separately
        self.n_gas = self.num_part[0]
        self.n_dark_matter = self.num_part[1]
        self.n_stars = self.num_part[4]
        self.n_black_holes = self.num_part[5]

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
            f"$\eta$ = {get_float('Kernel eta')} "
            rf"({get_float('Kernel target N_ngb')} $N_{{ngb}}$)"
            "\n"
            rf"$C_{{\rm CFL}}$ = {get_float('CFL parameter')}"
        )

        return output

    @property
    def viscosity_info(self):
        """
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


def generate_getter(filename, name: str, field: str, unit, mask: Union[None, np.array]):
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
                        setattr(
                            self, f"_{name}", unyt.unyt_array(handle[field][mask], unit)
                        )
                    else:
                        setattr(
                            self, f"_{name}", unyt.unyt_array(handle[field][...], unit)
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

    def __init__(self, filename, particle_type: int, units: SWIFTUnits):
        """
        This function primarily calls the generate_getters_for_particle_types
        function to ensure that we are reading correctly.
        """
        self.filename = filename

        self.particle_type = particle_type
        self.particle_name = self.particle_type_to_name(particle_type)

        self.generate_empty_properties()

        return

    def particle_type_to_name(self, particle_type: int):
        return metadata.particle_types.particle_name_underscores[particle_type]

    def generate_empty_properties(self):
        """
        Generates the empty properties that will be accessed through the
        setter and getters. We initially set all of the _{name} values
        to None. If it doesn't _exist_ in the file, we do not create the
        variable.
        """

        existing_fields = []

        with h5py.File(self.filename, "r") as handle:
            particle_handle = f"PartType{self.particle_type}"
            if particle_handle in handle:
                for key, name in getattr(
                    metadata.particle_fields, self.particle_name
                ).items():
                    if key in handle[particle_handle]:
                        existing_fields.append(name)

        for name in getattr(metadata.particle_fields, self.particle_name).values():
            if name in existing_fields:
                setattr(self, f"_{name}", None)

        return


def generate_dataset(filename, particle_type: int, units: SWIFTUnits, mask):
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

    particle_name = metadata.particle_types.particle_name_underscores[particle_type]
    particle_nice_name = metadata.particle_types.particle_name_class[particle_type]
    unit_system = getattr(units, particle_name)

    ThisDataset = type(
        f"{particle_nice_name}Dataset",
        __SWIFTParticleDataset.__bases__,
        dict(__SWIFTParticleDataset.__dict__),
    )

    existing_fields = []

    with h5py.File(filename, "r") as handle:
        particle_handle = f"PartType{particle_type}"
        if particle_handle in handle:
            for key, name in getattr(metadata.particle_fields, particle_name).items():
                if key in handle[particle_handle]:
                    existing_fields.append(name)

    for field_name, name in getattr(metadata.particle_fields, particle_name).items():
        if not name in existing_fields:
            continue

        unit = unit_system[name]

        if mask is not None:
            mask_array = getattr(mask, particle_name)
        else:
            mask_array = None

        setattr(
            ThisDataset,
            name,
            property(
                generate_getter(
                    filename,
                    name,
                    f"PartType{particle_type}/{field_name}",
                    unit=unit,
                    mask=mask_array,
                ),
                generate_setter(name),
                generate_deleter(name),
            ),
        )

    empty_dataset = ThisDataset(filename, particle_type, units)

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
        self.filename = filename
        self.mask = mask

        self.get_units()
        self.get_metadata()
        self.create_particle_datasets()

        return

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

        if not hasattr(self, "units"):
            self.get_units()

        for ptype, name in metadata.particle_types.particle_name_underscores.items():
            setattr(
                self,
                name,
                generate_dataset(self.filename, ptype, self.units, self.mask),
            )

        return
