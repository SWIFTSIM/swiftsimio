"""
Test for extra particle types.
"""
from swiftsimio import load, metadata
from swiftsimio import Writer
from swiftsimio.units import cosmo_units
import swiftsimio.metadata.particle as swp
import swiftsimio.metadata.writer.required_fields as swmw
import swiftsimio.metadata.unit.unit_fields as swuf
import swiftsimio.metadata.cosmology as swcf

import unyt
import numpy as np

import os

from copy import deepcopy


def generate_units(m, l, t, I, T):
    """
    This function is used to override the inbuilt swiftsimio generate_units function from
    metadata.unit.unit_fields. This allows the specification of a new particle type and 
       metadata.unit.unit_fields. This allows the specification of a new particle type and 
    metadata.unit.unit_fields. This allows the specification of a new particle type and 
    the values and types associated with that type.
    """
    dict_out = swuf.generate_units(m, l, t, I, T)

    extratype = {
        "coordinates": l,
        "masses": m,
        "particle_ids": None,
        "velocities": l / t,
        "potential": l * l / (t * t),
        "density": m / (l ** 3),
        "entropy": m * l ** 2 / (t ** 2 * T),
        "internal_energy": (l / t) ** 2,
        "smoothing_length": l,
        "pressure": m / (l * t ** 2),
        "diffusion": None,
        "sfr": m / t,
        "temperature": T,
        "viscosity": None,
        "specific_sfr": 1 / t,
        "material_id": None,
        "diffusion": None,
        "viscosity": None,
        "radiated_energy": m * (l / t) ** 2,
    }

    dict_out["extratype"] = extratype
    return dict_out


def generate_cosmology(scale_factor: float, gamma: float):
    """
    This function is used to override the inbuilt swiftsimio generate_cosmology function
    from metadata.cosmology. This allows the specification of a new particle type and 
    affects how the type is influenced by cosmology. Required only for reading in new
    particle types.
    """
    from swiftsimio.objects import cosmo_factor, a

    def cosmo_factory(a_dependence):
        return cosmo_factor(a_dependence, scale_factor)

    dict_out = swcf.generate_cosmology(scale_factor, gamma)
    no_cosmology = cosmo_factory(a / a)
    UNKNOWN = no_cosmology

    extratype = {**dict_out["gas"]}

    dict_out["extratype"] = extratype

    return dict_out


def test_write():
    """
    Tests whether swiftsimio can handle a new particle type. If the test doesn't crash
    this is a success.
    """
    # Use default units, i.e. cm, grams, seconds, Ampere, Kelvin
    unit_system = unyt.UnitSystem(
        name="default", length_unit=unyt.cm, mass_unit=unyt.g, time_unit=unyt.s
    )
    # Specify a new type in the metadata - currently done by editing the dictionaries directly.
    # TODO: Remove this terrible way of setting up different particle types.
    swp.particle_name_underscores[6] = "extratype"
    swp.particle_name_class[6] = "Extratype"
    swp.particle_name_text[6] = "Extratype"

    swmw.extratype = {"smoothing_length": "SmoothingLength", **swmw.shared}

    boxsize = 10 * unyt.cm

    x = Writer(unit_system, boxsize, unit_fields_generate_units=generate_units)

    x.extratype.coordinates = np.zeros((10, 3)) * unyt.cm
    for i in range(0, 10):
        x.extratype.coordinates[i][0] = float(i)

    x.extratype.velocities = np.zeros((10, 3)) * unyt.cm / unyt.s

    x.extratype.masses = np.ones(10, dtype=float) * unyt.g

    x.extratype.smoothing_length = np.ones(10, dtype=float) * (5.0 * unyt.cm)

    x.write("extra_test.hdf5")

    # Clean up these global variables we screwed around with...
    swp.particle_name_underscores.pop(6)
    swp.particle_name_class.pop(6)
    swp.particle_name_text.pop(6)


def test_read():
    """
    Tests whether swiftsimio can handle a new particle type. Has a few asserts to check the
    data is read in correctly.
    """
    swp.particle_name_underscores[6] = "extratype"
    swp.particle_name_class[6] = "Extratype"
    swp.particle_name_text[6] = "Extratype"

    swmw.extratype = {"smoothing_length": "SmoothingLength", **swmw.shared}

    metadata.particle_fields.extratype = {**metadata.particle_fields.gas}

    data = load("extra_test.hdf5")

    for i in range(0, 10):
        assert data.extratype.coordinates.value[i][0] == float(i)

    os.remove("extra_test.hdf5")

    # Clean up these global variables we screwed around with...
    swp.particle_name_underscores.pop(6)
    swp.particle_name_class.pop(6)
    swp.particle_name_text.pop(6)
