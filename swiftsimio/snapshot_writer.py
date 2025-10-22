"""Contains functions and objects for creating SWIFT datasets."""

import unyt
import h5py
import numpy as np

from typing import Callable
from sympy import Symbol
from functools import reduce

from swiftsimio import metadata
from swiftsimio.objects import cosmo_array
from swiftsimio.metadata.cosmology.cosmology_fields import a_exponents


def _ptype_str_to_int(ptype_str: str) -> int:
    """
    Convert a string like `"PartType0"` to an integer (in this example, `0`).

    Parameters
    ----------
    ptype_str : str
        The particle type string.

    Returns
    -------
    out : int
        The corresponding integer.
    """
    return int(ptype_str.strip("PartType")) if "PartType" in ptype_str else ptype_str


class __SWIFTWriterParticleDataset(object):
    """
    A particle dataset for _writing_ with.

    This is explicitly different to the one used for reading, as it requires a very
    different feature set. Perhaps one day they will be merged, but for now this keeps the
    code used to manage both simple.

    Parameters
    ----------
    unit_system : unyt.UnitSystem or str
        Either be a string (e.g. "cgs"), or a UnitSystem as defined by unyt
        specifying the units to be used. Users may wish to consider the
        cosmological unit system provided in swiftsimio.units.cosmo_units.

    particle_type : int
        The particle type of the dataset. Numbering convention is the same as
        SWIFT, with 0 corresponding to gas, etc. as usual.
    """

    def __init__(self, unit_system: unyt.UnitSystem | str, particle_type: int) -> None:
        self.unit_system = unit_system
        self.particle_type = particle_type

        self.particle_name = metadata.particle_types.particle_name_underscores[
            self.particle_type
        ]

        self.generate_empty_properties()

        return

    def generate_empty_properties(self) -> None:
        """
        Generate the empty properties accessed by setters and getters.

        Initially all of the _{name} values are set to None. Note that we
        only generate required properties.
        """
        for name in getattr(metadata.required_fields, self.particle_name).keys():
            setattr(self, f"_{name}", None)

        return

    def check_empty(self) -> bool:
        """
        Check if all required datasets are empty.

        Returns
        -------
        out : bool
            True if all datasets are empty.
        """
        for name in getattr(metadata.required_fields, self.particle_name).keys():
            if getattr(self, f"_{name}") is not None:
                return False

        return True

    def check_consistent(self) -> bool:
        """
        Perform consistency checks on dataset.

        Checks the following:

        + That all required fields (apart from particle_ids) are not None,
        + That all required fields have the same length

        If those are true,

        + self.n_part is set with the total number of particles of this type
        + self.requires_particle_ids_before_write is set to a boolean.

        Returns
        -------
        out : bool
            ``True`` if above listed criteria are satisfied.
        """
        self.requires_particle_ids_before_write = False

        sizes = []

        for name in getattr(metadata.required_fields, self.particle_name).keys():
            if getattr(self, f"_{name}") is None:
                if name == "particle_ids":
                    self.requires_particle_ids_before_write = True
                else:
                    raise AttributeError(f"Required dataset {name} is None.")
            else:
                sizes.append(getattr(self, f"_{name}").shape[0])

        # Now we figure out if everyone's the same (without numpy...)
        assert reduce(lambda x, y: x and y, [sizes[0] == x for x in sizes]), (
            f"Arrays passed to {self.particle_name} dataset are not of the same size."
        )

        # Make sure positions and velocities have the same shapes
        if getattr(self, "coordinates").shape != getattr(self, "velocities").shape:
            raise AttributeError(
                f"{self.particle_name} coordinates and velocities have unequal shapes"
            )

        self.n_part = sizes[0]

        return True

    def generate_smoothing_lengths(self, boxsize: cosmo_array, dimension: int) -> None:
        """
        Automatically generate smoothing lengths as 2 * the mean interparticle spacing.

        This only works for a uniform boxsize (i.e. one that has the same length in all
        dimensions). If boxsize is a list, we just use the 0th member.

        Parameters
        ----------
        boxsize : cosmo_array or cosmo_quantity
            Size of SWIFT computational box.

        dimension : int
            Number of box dimensions.
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

    def write_particle_group(self, file_handle: h5py.File, compress: bool) -> None:
        """
        Write the particle group's required properties to file.

        Parameters
        ----------
        file_handle : h5py.File
            File handle to write to.

        compress : bool
            Flag to indicate whether to turn on gzip compression.
        """
        particle_group = file_handle.create_group(self.particle_type)

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

    def write_particle_group_metadata(
        self, file_handle: h5py.File, dset_attributes: dict
    ) -> None:
        """
        Write the relevant metadata for a particle group.

        Parameters
        ----------
        file_handle : h5py.File
            HDF5 output file being written.

        dset_attributes : dict
            Dictionary containg metadata to attach to group.
        """
        for name, output_handle in getattr(
            metadata.required_fields, self.particle_name
        ).items():
            obj = file_handle[f"/{self.particle_type}/{output_handle}"]
            for attr_name, attr_value in dset_attributes[output_handle].items():
                obj.attrs.create(attr_name, attr_value)

        return

    def get_attributes(self, scale_factor: float) -> dict:
        """
        Return a dictionary containg the attributes to attach to the dataset.

        Parameters
        ----------
        scale_factor : float
            The cosmological scale factor of the dataset.

        Returns
        -------
        attributes_dict : dict
            Dictionary containg the attributes applying to the dataset.
        """
        attributes_dict = {}

        for name, output_handle in getattr(
            metadata.required_fields, self.particle_name
        ).items():
            field = getattr(self, name)

            # Find the exponents for each of the dimensions
            dim_exponents = get_dimensions(field.units.dimensions)

            # Find the scale factor associated quantities
            a_exp = a_exponents.get(name, 0)
            a_factor = scale_factor**a_exp

            attributes_dict[output_handle] = {
                "Conversion factor to CGS (not including cosmological corrections)": [
                    field.unit_quantity.in_cgs()
                ],
                "Conversion factor to physical CGS "
                "(including cosmological corrections)": [
                    field.unit_quantity.in_cgs() * a_factor
                ],
                # Assign the exponents in the proper order
                # (see unyt.dimensions.base_dimensions)
                "U_I exponent": [dim_exponents["(current)"]],
                "U_L exponent": [dim_exponents["(length)"]],
                "U_M exponent": [dim_exponents["(mass)"]],
                "U_T exponent": [dim_exponents["(temperature)"]],
                "U_t exponent": [dim_exponents["(time)"]],
                "a-scale exponent": [a_exp],
                "h-scale exponent": [0.0],
            }

        return attributes_dict


def get_dimensions(dimension: unyt.dimensions) -> dict:
    """
    Return exponents corresponding to base dimensions for given unyt dimensions object.

    Parameters
    ----------
    dimension : unyt.dimensions
        Dimension for which we're identifying the exponents.

    Returns
    -------
    exp_array : np.ndarray
        Array of exponents corresponding to each base dimension.

    Examples
    --------
    .. code-block:: python

        >>> get_dimensions(unyt.dimensions.velocity)
        {
            "(mass)": 0,
            "(length)": 1,
            "(time)": -1,
            "(temperature)": 0,
            "(current)": 0,
        }
    """
    # Get the names of all the dimensions
    dimensions = [str(x) for x in unyt.dimensions.base_dimensions[:5]]
    # Annoyingly it's current_mks instead of current in unyt, so change that
    dimensions[4] = "(current)"
    n_dims = len(dimensions)

    # create the return array
    exp_array = {}

    # extract the base and exponent for each of the units
    dim_array = [x.as_base_exp() for x in dimension.as_ordered_factors()]

    # Find out which dimensions and exponents are present in the base units
    for i in range(n_dims):
        present = False
        for dim in dim_array:
            if dimensions[i] in str(dim[0]):
                present = True
                exp_array[str(dim[0])] = float(dim[1])
        if not present:
            exp_array[dimensions[i]] = 0.0

    return exp_array


def generate_getter(
    name: str,
) -> Callable[[__SWIFTWriterParticleDataset], unyt.unyt_array]:
    """
    Generate a function that gets the unyt array for name.

    Parameters
    ----------
    name : str
        Name of data field.

    Returns
    -------
    getter : Callable
        Function that returns unyt array for ``name``.
    """

    def getter(self: __SWIFTWriterParticleDataset) -> unyt.unyt_array:
        """
        Get the value of the dataset from the private name attribute.

        Parameters
        ----------
        self : __SWIFTWriterParticleDataset
           The dataset the attribute is attached to.

        Returns
        -------
        out : unyt_array
            The value of the named data field.
        """
        return getattr(self, f"_{name}")

    return getter


def generate_setter(
    name: str, dimensions: unyt.dimensions, unit_system: unyt.UnitSystem | str
) -> Callable[[__SWIFTWriterParticleDataset, unyt.unyt_array], None]:
    """
    Generate a function that sets self._name to the value that is passed to it.

    Parameters
    ----------
    name : str
        String to set ``self._name`` to.

    dimensions : unyt.dimensions
        Physical dimension of ``self._name`` (for consistency check).

    unit_system : unyt.UnitSystem or str
        Unit system of ``self._name``.

    Returns
    -------
    out : Callable
        Function to set ``self._name``.
    """

    def setter(self: __SWIFTWriterParticleDataset, value: unyt.unyt_array) -> None:
        """
        Set the named dataset to a value (private name attribute).

        Parameters
        ----------
        self : __SWIFTWriterParticleDataset
            The dataset the attribute is attached to.

        value : unyt_array
            The value to set the attribute to.
        """
        if dimensions != 1:
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


def generate_deleter(name: str) -> Callable[[__SWIFTWriterParticleDataset], None]:
    """
    Generate a function that destroys self._name (sets it back to None).

    Parameters
    ----------
    name : str
        Name of object to be destroyed.

    Returns
    -------
    out : Callable
        Function to delete ``self._name``.
    """

    def deleter(self: __SWIFTWriterParticleDataset) -> None:
        """
        Delete the named dataset (private name attribute).

        Parameters
        ----------
        self : __SWIFTWriterParticleDataset
            The dataset the attribute is attached to.
        """
        current_value = getattr(self, f"_{name}")
        del current_value
        setattr(self, f"_{name}", None)

        return

    return deleter


def generate_dataset(
    unit_system: unyt.UnitSystem | str,
    particle_type: int,
    unit_fields_generate_units: Callable[
        ..., dict
    ] = metadata.unit_fields.generate_units,
) -> __SWIFTWriterParticleDataset:
    """
    Generate a SWIFTWriterParticleDataset _class_ for the given particle type.

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
        Unit system of the dataset.

    particle_type : int
        The particle type of the dataset. Numbering convention is the same as
        SWIFT, with 0 corresponding to gas, etc. as usual.

    unit_fields_generate_units : Callable, optional
        Collection of properties in metadata file for which to create setters
        and getters.

    Returns
    -------
    SWIFTWriterParticleDataset
        An empty dataset for the given particle type.
    """
    particle_name = metadata.particle_types.particle_name_underscores[particle_type]
    particle_nice_name = metadata.particle_types.particle_name_class[particle_type]

    this_dataset_bases = (__SWIFTWriterParticleDataset, object)
    this_dataset_dict = {}

    # Get the unit dimensions
    dimensions = metadata.unit_fields.generate_dimensions(unit_fields_generate_units)

    for name in getattr(metadata.required_fields, particle_name).keys():
        this_dataset_dict[name] = property(
            generate_getter(name),
            generate_setter(name, dimensions[particle_name][name], unit_system),
            generate_deleter(name),
        )

    ThisDataset = type(
        f"{particle_nice_name}WriterDataset", this_dataset_bases, this_dataset_dict
    )

    empty_dataset = ThisDataset(unit_system, particle_type)

    return empty_dataset


class SWIFTSnapshotWriter(object):
    """
    The SWIFT dataset writer.

    This is used to store all particle arrays and do some extra processing before writing
    a HDF5 file containing:

    + Fully consistent unit system
    + All required arrays for SWIFT to start
    + Required metadata (all automatic, apart from those required by __init__)

    Parameters
    ----------
    unit_system : unyt.UnitSystem or str
        Unit system for dataset.

    boxsize : cosmo_array
        Size of simulation box and associated units.

    dimension : int, optional
        Dimensions of simulation.

    compress : bool, optional
        Flag to turn on compression.

    extra_header : dict, optional
        Dictionary containing extra things to write to the header.

    unit_fields_generate_units : Callable, optional
        Collection of properties in metadata file for which to create setters
        and getters.

    scale_factor : np.float32
        Scale factor associated with dataset. Defaults to 1.
    """

    def __init__(
        self,
        unit_system: unyt.UnitSystem | str,
        boxsize: cosmo_array,
        dimension: int = 3,
        compress: bool = True,
        extra_header: dict | None = None,
        unit_fields_generate_units: Callable[
            ..., dict
        ] = metadata.unit_fields.generate_units,
        scale_factor: np.float32 = 1.0,
    ) -> None:
        self.unit_fields_generate_units = unit_fields_generate_units
        if isinstance(unit_system, str):
            self.unit_system = unyt.unit_systems.unit_system_registry[unit_system]
        else:
            self.unit_system = unit_system

        # Validate the boxsize and convert to our units.
        try:
            for x in boxsize:
                x.convert_to_base(self.unit_system)
            self.boxsize = boxsize
        except TypeError:
            # This is just a single number (i.e. uniform in all dimensions)
            boxsize.convert_to_base(self.unit_system)
            self.boxsize = boxsize

        self.dimension = dimension
        self.compress = compress

        self.extra_header = extra_header

        self.create_particle_datasets()

        self.scale_factor = scale_factor

        return

    def create_particle_datasets(self) -> None:
        """Create particle dataset for each particle type in the metadata."""
        for number, name in metadata.particle_types.particle_name_underscores.items():
            setattr(
                self,
                name,
                generate_dataset(
                    self.unit_system, number, self.unit_fields_generate_units
                ),
            )

        return

    def _generate_ids(self, names_to_write: list) -> None:
        """
        (Re-)generate all particle IDs for groups with names in names_to_write.

        Parameters
        ----------
        names_to_write : list
            List of groups to regenerate ids for.
        """
        numbers_of_particles = [getattr(self, name).n_part for name in names_to_write]
        # Start particle ID's at 1. When running with hydro + DM, partID = 0
        # is a no-no because the ID's are used as offsets in arrays. The code
        # will most likely crash at some point, and "part_verify_links()" will
        # complain about it if SWIFT is run with debugging checks on.
        already_used = 1

        for number, name in zip(numbers_of_particles, names_to_write):
            getattr(self, name).particle_ids = np.arange(
                already_used, number + already_used
            )
            already_used += number

        return

    def _write_metadata(self, handle: h5py.File, names_to_write: list) -> None:
        """
        Write metadata to file.

        Metadata written is based on the information passed to the object and the
        information in the particle groups.

        Parameters
        ----------
        handle : h5py.File
            HDF5 file handle to write to.
        names_to_write : list
            List of metadata fields to write.
        """
        part_types = (
            max(
                [
                    _ptype_str_to_int(k)
                    for k in metadata.particle.particle_name_underscores.keys()
                ]
            )
            + 1
        )
        number_of_particles = [0] * part_types
        mass_table = [0.0] * part_types

        for number, name in metadata.particle_types.particle_name_underscores.items():
            if name in names_to_write:
                number_of_particles[_ptype_str_to_int(number)] = getattr(
                    self, name
                ).n_part
                mass_table[_ptype_str_to_int(number)] = getattr(self, name).masses[0]

        attrs = {
            "BoxSize": self.boxsize,
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

    def _write_units(self, handle: h5py.File) -> None:
        """
        Write the unit information to file.

        Note that we do not have support for unit current yet.

        Parameters
        ----------
        handle : h5py.File
            HDF5 file to write units to.
        """
        dim = unyt.dimensions
        cgs_base = unyt.unit_systems.cgs_unit_system.base_units
        base = self.unit_system.base_units

        def get_conversion(unit_type: Symbol) -> np.ndarray[float]:
            """
            Find the conversion factor from base to cgs units.

            Parameters
            ----------
            unit_type : sympy.Symbol
                The type of unit to convert (mass, length, etc.).

            Returns
            -------
            out : np.ndarray[float]
                The conversion factor.
            """
            # We need to find the correct unit (which is now stored as a sympy value,
            # why?!) and convert it to an unyt unit.
            our_unit = unyt.unit_object.Unit(base[unit_type])
            cgs_unit = unyt.unit_object.Unit(cgs_base[unit_type])
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

    def write(self, filename: str) -> None:
        """
        Write the information in the dataset to file.

        Parameters
        ----------
        filename : str
            File to write to.
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
