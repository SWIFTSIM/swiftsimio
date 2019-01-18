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


class SWIFTUnits(object):
    """
    Generates a unyt system that can then be used with the SWIFT data.
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

    def generate_units(self):
        """
        Creates the unyt system to use to reduce data.
        """

        unit_fields = metadata.unit_fields.generate_units(
            m=self.units["Unit mass in cgs (U_M)"],
            l=self.units["Unit length in cgs (U_L)"],
            t=self.units["Unit time in cgs (U_t)"],
            I=self.units["Unit current in cgs (U_I)"],
            T=self.units["Unit temperature in cgs (U_T)"]
        )

        for ptype, units in unit_fields.items():
            setattr(self, ptype, units)

        return


class SWIFTMetadata(object):
    def __init__(self):
        pass


def generate_getter(filename, name: str, field: str, unit):
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
                    setattr(self, f"_{name}",
                        unyt.unyt_array(handle[field][...], unit)
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
        to None.
        """

        for name in getattr(metadata.particle_fields, self.particle_name).values():
            setattr(self, f"_{name}", None)

        return


def generate_dataset(filename, particle_type: int, units: SWIFTUnits):
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

    ThisDataset = type(f"{particle_nice_name}Dataset", __SWIFTParticleDataset.__bases__, dict(__SWIFTParticleDataset.__dict__))


    for field_name, name in getattr(metadata.particle_fields, particle_name).items():
        unit = unit_system[name]

        setattr(
            ThisDataset,
            name,
            property(
                generate_getter(
                    filename, name, f"PartType{particle_type}/{field_name}", unit=unit
                ),
                generate_setter(name),
                generate_deleter(name),
            ),
        )

    empty_dataset = ThisDataset(filename, particle_type, units)

    return empty_dataset


class SWIFTDataset(object):
    def __init__(self, filename):
        self.filename = filename

        self.get_units()
        self.get_metadata()
        self.create_particle_datasets()
        pass

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

        self.metadata = SWIFTMetadata()

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
            setattr(self, name, generate_dataset(self.filename, ptype, self.units))

        return
