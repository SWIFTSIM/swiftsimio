"""
Define units "by hand" for writing snapshot-like files.

Contains the information for the units that determine the particle fields.
When reading files we read this metadata from the files, but when writing
we may need to generate this information.
"""

from unyt import g, cm, s, statA, K, Unit
from typing import Callable

# scale factor exponents for writing snapshot-like files:
a_exponents = {"coordinates": 1, "internal_energies": -2}


def generate_units(
    mass: Unit, length: Unit, time: Unit, current: Unit, temperature: Unit
) -> dict[str, dict[str, Unit]]:
    """
    Generate unit dictionaries.

    Units for:

    mass, length, time, current, and temperature

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
    dict[str, dict[str, Unit]]
        Dictionary with a dictonary for each particle type defining the units for each
        field.
    """
    shared = {
        "coordinates": length,
        "masses": mass,
        "particle_ids": None,
        "velocities": length / time,
        "potential": length * length / (time * time),
    }

    gas = {
        "internal_energy": (length / time) ** 2,
        "smoothing_length": length,
        "pressure": mass / (length * time**2),
        "temperature": temperature,
        **shared,
    }

    dark_matter = {**shared}

    boundary = {**shared}

    sinks = {**shared}

    stars = {
        "smoothing_length": length,
        **shared,
    }

    black_holes = {**shared}

    neutrinos = {**shared}

    return {
        "gas": gas,
        "dark_matter": dark_matter,
        "boundary": boundary,
        "sinks": sinks,
        "stars": stars,
        "black_holes": black_holes,
        "neutrinos": neutrinos,
    }


def generate_dimensions(
    generate_unit_func: Callable[..., dict] = generate_units,
) -> dict:
    """
    Get the dimensions for the above.

    Parameters
    ----------
    generate_unit_func : Callable[..., dict]
        Function to set units for particle fields.

    Returns
    -------
    dict
        Dictionary specifying dimensions.

    """
    units = generate_unit_func(g, cm, s, statA, K)

    dimensions = {}

    for particle_type, particle_type_units in units.items():
        dimensions[particle_type] = {}

        for unit_name, unit in particle_type_units.items():
            try:
                dimensions[particle_type][unit_name] = unit.dimensions
            except AttributeError:
                # Units that have "none" dimensions
                dimensions[particle_type][unit_name] = 1

    return dimensions
