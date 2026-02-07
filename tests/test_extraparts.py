"""Test for extra particle types."""

from swiftsimio import load, metadata, Writer, cosmo_array
import swiftsimio.metadata.particle as swp
import swiftsimio.metadata.writer.required_fields as swmw
import swiftsimio.metadata.unit.unit_fields as swuf

import unyt
from unyt import Unit
import numpy as np

import os


def generate_units(
    mass: Unit, length: Unit, time: Unit, current: Unit, temperature: Unit
) -> dict[str, dict[str, Unit]]:
    """
    Generate units differently for testing.

    This function is used to override the inbuilt swiftsimio generate_units function from
    metadata.unit.unit_fields. This allows the specification of a new particle type and
    the values and types associated with that type.

    Parameters
    ----------
    mass : Unit
        The mass unit.

    length : Unit
        The length unit.

    time : Unit
        The time unit.

    current : Unit
        The current unit.

    temperature : Unit
        The temperature unit.

    Returns
    -------
    dict[str, Unit]
        A dictionary mapping field names to units.
    """
    dict_out = swuf.generate_units(mass, length, time, current, temperature)

    extratype = {
        "coordinates": length,
        "masses": mass,
        "particle_ids": None,
        "velocities": length / time,
        "potential": length * length / (time * time),
        "density": mass / (length**3),
        "entropy": mass * length**2 / (time**2 * temperature),
        "internal_energy": (length / time) ** 2,
        "smoothing_length": length,
        "pressure": mass / (length * time**2),
        "diffusion": None,
        "sfr": mass / time,
        "temperature": temperature,
        "viscosity": None,
        "specific_sfr": 1 / time,
        "material_id": None,
        "radiated_energy": mass * (length / time) ** 2,
    }

    dict_out["extratype"] = extratype
    return dict_out


def test_write():
    """
    Tests whether swiftsimio can handle a new particle type.

    If the test doesn't crash this is a success.
    """
    # Use default units, i.e. cm, grams, seconds, Ampere, Kelvin
    unit_system = unyt.UnitSystem(
        name="default", length_unit=unyt.cm, mass_unit=unyt.g, time_unit=unyt.s
    )
    # Specify a new type in the metadata - currently done by editing the dictionaries
    # directly.
    # TODO: Remove this terrible way of setting up different particle types.
    swp.particle_name_underscores["PartType7"] = "extratype"
    swp.particle_name_class["PartType7"] = "Extratype"
    swp.particle_name_text["PartType7"] = "Extratype"

    swmw.extratype = {"smoothing_length": "SmoothingLength", **swmw._shared}

    a = 0.5
    boxsize = cosmo_array(
        [10, 10, 10], unyt.cm, comoving=False, scale_factor=a, scale_exponent=1
    )

    x = Writer(
        unit_system, boxsize, unit_fields_generate_units=generate_units, scale_factor=a
    )

    x.extratype.coordinates = cosmo_array(
        np.array([np.arange(10), np.zeros(10), np.zeros(10)]).astype(float).T,
        unyt.cm,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=1,
    )
    x.extratype.velocities = cosmo_array(
        np.zeros((10, 3), dtype=float),
        unyt.cm / unyt.s,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=0,
    )

    x.extratype.masses = cosmo_array(
        np.ones(10, dtype=float),
        unyt.g,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=0,
    )

    x.extratype.smoothing_length = cosmo_array(
        np.ones(10, dtype=float) * 5.0,
        unyt.cm,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=1,
    )

    x.write("extra_test.hdf5")

    # Clean up these global variables we screwed around with...
    swp.particle_name_underscores.pop("PartType7")
    swp.particle_name_class.pop("PartType7")
    swp.particle_name_text.pop("PartType7")


def test_read():
    """
    Test whether swiftsimio can handle a new particle type.

    Has a few asserts to check the data is read in correctly.
    """
    swp.particle_name_underscores["PartType7"] = "extratype"
    swp.particle_name_class["PartType7"] = "Extratype"
    swp.particle_name_text["PartType7"] = "Extratype"

    swmw.extratype = {"smoothing_length": "SmoothingLength", **swmw._shared}

    metadata.particle_fields.extratype = {**metadata.particle_fields.gas}

    data = load("extra_test.hdf5")

    for i in range(0, 10):
        assert data.extratype.coordinates.value[i][0] == float(i)

    os.remove("extra_test.hdf5")

    # Clean up these global variables we screwed around with...
    swp.particle_name_underscores.pop("PartType7")
    swp.particle_name_class.pop("PartType7")
    swp.particle_name_text.pop("PartType7")
