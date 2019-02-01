"""
Contains functions and objects for creating SWIFT datasets.
"""

import unyt

from typing import Union

from swiftsimio import metadata

class __SWIFTWriterParticleDataset(object):
    """
    A particle dataset for _writing_ with. This is explicitly different
    to the one used for reading, as it requires a very different feature
    set. Perhaps one day they will be merged, but for now this keeps the
    code used to manage both simple.
    """
    def __init__(self, unit_system: Union[unyt.UnitSystem, str], particle_type: int):
        """
        Takes the unit system as a parameter. This can either be a string (e.g. "cgs"),
        or a UnitSystem as defined by unyt. Users may wish to consider the cosmological
        unit system provided in swiftsimio.units.cosmo_units.

        The other parameter is the particle type, with 0 corresponding to gas, etc.
        as usual.
        """

        self.unit_system = unit_system
        self.particle_type = particle_type

        self.particle_handle = f"PartType{self.particle_type}"
        self.particle_name = metadata.particle_types.particle_name_underscores[self.particle_type]

        self.generate_empty_properties()
        
        return

    def generate_empty_properties(self):
        """
        Generates the empty properties that will be accessed through the
        setter and getters. We initially set all of the _{name} values
        to None. Note that we only generate required properties.
        """

        for name in getattr(metadata.required_fields, self.particle_name).values():
            setattr(self, f"_{name}", None)

        return

