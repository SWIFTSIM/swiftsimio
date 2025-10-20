"""Test for extra particle types."""

from sympy import Expr
from swiftsimio import load, metadata, Writer
from swiftsimio.objects import cosmo_factor
import swiftsimio.metadata.particle as swp
import swiftsimio.metadata.writer.required_fields as swmw
import swiftsimio.metadata.unit.unit_fields as swuf
import swiftsimio.metadata.cosmology as swcf

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
       metadata.unit.unit_fields. This allows the specification of a new particle type and
    metadata.unit.unit_fields. This allows the specification of a new particle type and
    the values and types associated with that type.
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


def generate_cosmology(
    scale_factor: float, gamma: float
) -> dict[str, dict[str, cosmo_factor]]:
    """
    Generate cosmology differently for testing.

    This function is used to override the inbuilt swiftsimio generate_cosmology function
    from metadata.cosmology. This allows the specification of a new particle type and
    affects how the type is influenced by cosmology. Required only for reading in new
    particle types.
    """

    def cosmo_factory(a_dependence: Expr) -> cosmo_factor:
        """
        Generate a ``cosmo_factor``.

        Parameters
        ----------
        a_dependence : Expr
            The scale factor dependence desired.

        Returns
        -------
        out : cosmo_factor
            One of our ``cosmo_factor`` objects.
        """
        return cosmo_factor(a_dependence, scale_factor)

    dict_out = swcf.generate_cosmology(scale_factor, gamma)

    extratype = {**dict_out["gas"]}

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
