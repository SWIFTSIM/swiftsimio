"""
Contains functions and objects for creating SWIFT datasets.

Essentially all you want to do is use SWIFTWriterDataset and fill the attributes
that are required for each particle type. More information is available in the
README.
"""

import unyt
import h5py
import numpy as np

from typing import Union, List, Callable
from functools import reduce

from swiftsimio import metadata
from re import findall


class __SWIFTWriterParticleDataset(object):
    """
    A particle dataset for _writing_ with. This is explicitly different
    to the one used for reading, as it requires a very different feature
    set. Perhaps one day they will be merged, but for now this keeps the
    code used to manage both simple.

    Methods
    -------

    generate_empty_properties(self)
        generates the empty properties that will be accessed through the
        setter and getters. 
    check_empty(self) 
        checks if all required datasets are empty.
    check_consistent(self) 
        performs consistency checks on dataset
    generate_smoothing_lengths(self, boxsize: Union[List[unyt.unyt_quantity], unyt.unyt_quantity], dimension: int)
        automatically generates the smoothing lengths 
    write_particle_group(self, file_handle: h5py.File, compress: bool)
        writes the particle group's required properties to file.
    """

    def __init__(self, unit_system: Union[unyt.UnitSystem, str], particle_type: int):
        """
        Generates the requested unit system

        Parameters
        ----------
        unit_system : unyt.UnitSystem or str
            either be a string (e.g. "cgs"), or a UnitSystem as defined by unyt 
            specifying the units to be used. Users may wish to consider the 
            cosmological unit system provided in swiftsimio.units.cosmo_units.

        particle_type : int
            the particle type of the dataset. Numbering convention is the same as 
            SWIFT, with 0 corresponding to gas, etc.  as usual.
        """

        self.unit_system = unit_system
        self.particle_type = particle_type

        self.particle_handle = f"PartType{self.particle_type}"
        self.particle_name = metadata.particle_types.particle_name_underscores[
            self.particle_type
        ]

        self.generate_empty_properties()

        return

    def generate_empty_properties(self):
        """
        Generates the empty properties that will be accessed through the
        setter and getters. 
        
        Initially all of the _{name} values are set to None. Note that we 
        only generate required properties.
        """

        for name in getattr(metadata.required_fields, self.particle_name).keys():
            setattr(self, f"_{name}", None)

        return

    def check_empty(self) -> bool:
        """
        Checks if all required datasets are empty.

        Returns
        -------
        bool
            True if all datasets are empty
        """

        for name in getattr(metadata.required_fields, self.particle_name).keys():
            if getattr(self, f"_{name}") is not None:
                return False

        return True

    def check_consistent(self) -> bool:
        """
        Performs consistency checks on dataset

        Checks the following:

        + That all required fields (apart from particle_ids) are not None,
        + That all required fields have the same length

        If those are true,

        + self.n_part is set with the total number of particles of this type
        + self.requires_particle_ids_before_write is set to a boolean.

        Returns
        -------
        bool
            True if above listed criteria are satisfied
        """

        self.requires_particle_ids_before_write = False

        sizes = []

        for name in getattr(metadata.required_fields, self.particle_name).keys():
            if getattr(self, f"_{name}") is None:
                if name is "particle_ids":
                    self.requires_particle_ids_before_write = True
                else:
                    raise AttributeError(f"Required dataset {name} is None.")
            else:
                sizes.append(getattr(self, f"_{name}").shape[0])

        # Now we figure out if everyone's the same (without numpy...)
        assert reduce(
            lambda x, y: x and y, [sizes[0] == x for x in sizes]
        ), f"Arrays passed to {self.particle_name} dataset are not of the same size."

        self.n_part = sizes[0]

        return True

    def generate_smoothing_lengths(
        self,
        boxsize: Union[List[unyt.unyt_quantity], unyt.unyt_quantity],
        dimension: int,
    ):
        """
        Automatically generates the smoothing lengths as 2 * the mean interparticle separation.

        This only works for a uniform boxsize (i.e. one that has the same length in all dimensions).
        If boxsize is a list, we just use the 0th member.

        Parameters
        ----------
        boxsize : unyt.unyt_quantity or list of unyt.unyt_quantity
            size of SWIFT computational box
        dimension : int
            number of box dimensions
        """

        try:
            boxsize = boxsize[0]
        except IndexError:
            boxsize = boxsize

        n_part = self.coordinates.shape[0]
        mips = boxsize / (n_part ** (1.0 / dimension))

        smoothing_lengths = mips * np.ones(n_part, dtype=float)

        self.smoothing_length = smoothing_lengths

        return

    def write_particle_group(self, file_handle: h5py.File, compress: bool):
        """
        Writes the particle group's required properties to file.

        Parameters
        ----------
        file_handle : h5py.File
            file handle to write to
        compress : bool
            flag to indicate whether to turn on gzip compression
        """

        particle_group = file_handle.create_group(self.particle_handle)

        if compress:
            compression = "gzip"
        else:
            compression = None

        for name, output_handle in getattr(
            metadata.required_fields, self.particle_name
        ).items():
            particle_group.create_dataset(
                output_handle, data=getattr(self, name), compression=compression
            )

        return

    def write_particle_group_metadata(self, file_handle: h5py.File, dset_attributes):
        for name, output_handle in getattr(
            metadata.required_fields, self.particle_name
        ).items():
            obj = file_handle[f"/PartType{self.particle_type}/{output_handle}"]
            for attr_name, attr_value in dset_attributes[output_handle].items():
                obj.attrs.create(attr_name, attr_value)

        return

    def get_attributes(self, scale_factor):
        attributes_dict = {}
        a_exp_dict = {
            "coordinates": 1,
            "internal_energies": -2,
        }

        for name, output_handle in getattr(
            metadata.required_fields, self.particle_name
        ).items():
            field = getattr(self, name)

            # Find the exponents for each of the dimensions
            dim_exponents = get_dimensions(field.units.dimensions)

            # Find the scale factor associated quantities
            a_exp = a_exp_dict.get(name, 0)
            a_factor = scale_factor * a_exp

            attributes_dict[output_handle] = {
                "Conversion factor to CGS (not including cosmological corrections)": [
                    field.unit_quantity.in_cgs()
                ],
                "Conversion factor to physical CGS (including cosmological corrections)": [
                    field.unit_quantity.in_cgs() * a_factor
                ],
                "Description": b"Co-moving positions of the particles",
                "Expression for physical CGS units": b"a U_L  [ cm ]",
                "U_I exponent": [dim_exponents[0]],
                "U_L exponent": [dim_exponents[1]],
                "U_M exponent": [dim_exponents[2]],
                "U_T exponent": [dim_exponents[3]],
                "U_t exponent": [dim_exponents[4]],
                "a-scale exponent": [a_exp],
                "h-scale exponent": [0.0],
            }

        return attributes_dict


def get_dimensions(obj):
    dimensions = ["current", "length", "mass", "temperature", "time"]
    n_dims = len(dimensions)
    exp_array = np.zeros(n_dims, dtype=np.float32)
    dim_array = [x.as_base_exp() for x in obj.as_ordered_factors()]

    for i in range(n_dims):
        for dim in dim_array:
            if dimensions[i] in str(dim[0]):
                exp_array[i] = dim[1]

    return exp_array


def generate_getter(name: str):
    """
    Generates a function that gets the unyt array for name.

    Parameters
    ----------
    name : str
        name of data field

    Returns
    -------
    getter : function
        function that returns unyt array for `name`
    """

    def getter(self):
        return getattr(self, f"_{name}")

    return getter


def generate_setter(name: str, dimensions, unit_system: Union[unyt.UnitSystem, str]):
    """
    Generates a function that sets self._name to the value that is passed to it.
    
    Parameters
    ----------
    name : str
        string to set self._name to
    dimensions : unyt.dimensions
        physical dimension of self._name (for consistency check)
    unit_system : unyt.UnitSystem or str
        unit system of self._name

    Returns
    -------
    setter : function
        function to set self._name
    """

    def setter(self, value: unyt.unyt_array):
        if dimensions is not 1:
            if isinstance(value, unyt.unyt_array):
                if value.units.dimensions == dimensions:
                    value.convert_to_base(unit_system)

                    setattr(self, f"_{name}", value)
                else:
                    raise unyt.exceptions.InvalidUnitEquivalence(
                        f"Convert to {name}", value.units.dimensions, dimensions
                    )
            else:
                raise TypeError("You must provide quantities as unyt arrays.")
        else:
            setattr(self, f"_{name}", unyt.unyt_array(value, None))

        return

    return setter


def generate_deleter(name: str):
    """
    Generates a function that destroys self._name (sets it back to None).

    Parameters
    ----------
    name : str
        name of object to be destroyed

    Returns
    -------
    deleter : function
        function to delete self._name
    """

    def deleter(self):
        current_value = getattr(self, f"_{name}")
        del current_value
        setattr(self, f"_{name}", None)

        return

    return deleter


def generate_dataset(
    unit_system: Union[unyt.UnitSystem, str],
    particle_type: int,
    unit_fields_generate_units: Callable[
        ..., dict
    ] = metadata.unit_fields.generate_units,
):
    """
    Generates a SWIFTWriterParticleDataset _class_ that corresponds to the
    particle type given.

    We _must_ do the following _outside_ of the class itself, as one
    can assign properties to a _class_ but not _within_ a class
    dynamically.

    Here we loop through all of the possible properties in the metadata file.
    We then use the builtin property() function and some generators to
    create setters and getters for those properties. This will allow them
    to be accessed from outside by using SWIFTWriterParticleDataset.name, where
    the name is, for example, coordinates.

    Parameters
    ----------
    unit_system : unyt.UnitSystem or str
        unit system of the dataset
    particle_type : int
        the particle type of the dataset. Numbering convention is the same as 
        SWIFT, with 0 corresponding to gas, etc.  as usual.
    unit_fields_generate_units : callable, optional
        collection of properties in metadata file for which to create setters
        and getters

    Returns
    -------
    SWIFTWriterParticleDataset
        an empty dataset for the given particle type
    """

    particle_name = metadata.particle_types.particle_name_underscores[particle_type]
    particle_nice_name = metadata.particle_types.particle_name_class[particle_type]

    ThisDataset = type(
        f"{particle_nice_name}WriterDataset",
        __SWIFTWriterParticleDataset.__bases__,
        dict(__SWIFTWriterParticleDataset.__dict__),
    )

    # Get the unit dimensions
    dimensions = metadata.unit_fields.generate_dimensions(unit_fields_generate_units)

    for name in getattr(metadata.required_fields, particle_name).keys():
        setattr(
            ThisDataset,
            name,
            property(
                generate_getter(name),
                generate_setter(name, dimensions[particle_name][name], unit_system),
                generate_deleter(name),
            ),
        )

    empty_dataset = ThisDataset(unit_system, particle_type)

    return empty_dataset


class SWIFTWriterDataset(object):
    """
    The SWIFT writer dataset. This is used to store all particle arrays and do
    some extra processing before writing a HDF5 file containing:

    + Fully consistent unit system
    + All required arrays for SWIFT to start
    + Required metadata (all automatic, apart from those required by __init__)
    """

    def __init__(
        self,
        unit_system: Union[unyt.UnitSystem, str],
        box_size: Union[list, unyt.unyt_quantity],
        dimension=3,
        compress=True,
        extra_header: Union[None, dict] = None,
        unit_fields_generate_units: Callable[
            ..., dict
        ] = metadata.unit_fields.generate_units,
        scale_factor: np.float32 = 1.0,
    ):
        """
        Creates SWIFTWriterDataset object

        Parameters
        ----------
        unit_system : unyt.UnitSystem or str
            unit system for dataset
        boxsize : list or unyt.unyt_quantity
            size of simulation box and associated units
        dimension : int, optional
            dimensions of simulation
        compress : bool, optional
            flag to turn on compression
        extra_header : dict, optional
            dictionary containing extra things to write to the header
        unit_fields_generate_units: callable, optional
            collection of properties in metadata file for which to create setters
            and getters
        scale_factor: np.float32
            scale factor associated with dataset. Defaults to 1
            
        """
        self.unit_fields_generate_units = unit_fields_generate_units
        if isinstance(unit_system, str):
            self.unit_system = unyt.unit_systems.unit_system_registry[unit_system]
        else:
            self.unit_system = unit_system

        # Validate the boxsize and convert to our units.
        try:
            for x in box_size:
                x.convert_to_base(self.unit_system)
            self.box_size = box_size
        except TypeError:
            # This is just a single number (i.e. uniform in all dimensions)
            box_size.convert_to_base(self.unit_system)
            self.box_size = box_size

        self.dimension = dimension
        self.compress = compress

        self.extra_header = extra_header

        self.create_particle_datasets()

        self.scale_factor = scale_factor

        return

    def create_particle_datasets(self):
        """
        Creates particle dataset for each particle type in the metadata with 
        associated units
        """
        for number, name in metadata.particle_types.particle_name_underscores.items():
            setattr(
                self,
                name,
                generate_dataset(
                    self.unit_system, number, self.unit_fields_generate_units
                ),
            )

        return

    def _generate_ids(self, names_to_write: List):
        """
        (Re-)generates all particle IDs for groups with names in names_to_write.

        Parameters
        ----------
        names_to_write : list
            list of groups to regenerate ids for
        """

        numbers_of_particles = [getattr(self, name).n_part for name in names_to_write]
        already_used = 0

        for number, name in zip(numbers_of_particles, names_to_write):
            getattr(self, name).particle_ids = np.arange(
                already_used, number + already_used
            )
            already_used += number

        return

    def _write_metadata(self, handle: h5py.File, names_to_write: List):
        """
        Writes metadata to file based on the information passed to the object
        and the information in the particle groups.

        Parameters
        ----------
        handle : h5py.File
            hdf5 file handle to write to
        names_to_write : list
            list of metadata fields to write
        """
        part_types = max(metadata.particle.particle_name_underscores.keys()) + 1
        number_of_particles = [0] * part_types
        mass_table = [0.0] * part_types

        for number, name in metadata.particle_types.particle_name_underscores.items():
            if name in names_to_write:
                number_of_particles[number] = getattr(self, name).n_part
                mass_table[number] = getattr(self, name).masses[0]

        attrs = {
            "BoxSize": self.box_size,
            "NumPart_Total": number_of_particles,
            "NumPart_Total_HighWord": [0] * 6,
            "Flag_Entropy_ICs": 0,
            "Dimension": np.array([self.dimension]),
            # LEGACY but required for Gadget readers
            "NumFilesPerSnapshot": 1,
            "NumPart_ThisFile": number_of_particles,
            "MassTable": mass_table,
        }

        if self.extra_header is not None:
            attrs = {**attrs, **self.extra_header}

        header = handle.create_group("Header")

        for name, value in attrs.items():
            header.attrs.create(name, value)

        return

    def _write_units(self, handle: h5py.File):
        """
        Writes the unit information to file.

        Note that we do not have support for unit current yet.

        Parameters
        ----------
        handle : h5py.File
            hdf5 file to write units to
        """

        dim = unyt.dimensions
        cgs_base = unyt.unit_systems.cgs_unit_system.base_units
        base = self.unit_system.base_units

        def get_conversion(type):
            # We need to find the correct unit (which is now stored as a sympy value,
            # why?!) and convert it to an unyt unit.
            our_unit = unyt.unit_object.Unit(base[type])
            cgs_unit = unyt.unit_object.Unit(cgs_base[type])
            conversion_factor = our_unit.get_conversion_factor(cgs_unit)[0]

            # We use the array because this is how swift outputs it, as a length
            # 1 array (rather than as a single float).
            return np.array([conversion_factor])

        attrs = {
            "Unit mass in cgs (U_M)": get_conversion(dim.mass),
            "Unit length in cgs (U_L)": get_conversion(dim.length),
            "Unit time in cgs (U_t)": get_conversion(dim.time),
            "Unit current in cgs (U_I)": np.array([1.0]),
            "Unit temperature in cgs (U_T)": get_conversion(dim.temperature),
        }

        units = handle.create_group("Units")

        for name, value in attrs.items():
            units.attrs.create(name, value)

        return

    def write(self, filename: str):
        """
        Writes the information in the dataset to file.

        Parameters
        ----------
        filename : str
            file to write to
        """

        names_to_write = []
        generate_ids = False

        for name in metadata.particle_types.particle_name_underscores.values():
            this_dataset = getattr(self, name)

            if not this_dataset.check_empty():
                if this_dataset.check_consistent():
                    names_to_write.append(name)
                    generate_ids = (
                        generate_ids or this_dataset.requires_particle_ids_before_write
                    )

        if generate_ids:
            self._generate_ids(names_to_write)

        # Now we do the hard part
        with h5py.File(filename, "w") as handle:
            self._write_metadata(handle, names_to_write)

            self._write_units(handle)

            for name in names_to_write:
                getattr(self, name).write_particle_group(handle, compress=self.compress)
                attrs = getattr(self, name).get_attributes(self.scale_factor)
                getattr(self, name).write_particle_group_metadata(handle, attrs)

        return
