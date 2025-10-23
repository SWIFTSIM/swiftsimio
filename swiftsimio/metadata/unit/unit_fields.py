"""
Deprecated: define units "by hand" instead of reading them from files.

Contains the information for the units that determine the
particle fields. This must be provided, if not swiftsimio
will crash (as it should, you can'time just be going around
having quantities without units...).

Unfortunately there must be a generator function used here
because we don'time know the units ahead of time.
"""

from unyt import g, cm, s, statA, K, Unit
from typing import Callable


def generate_units(
    mass: Unit, length: Unit, time: Unit, current: Unit, temperature: Unit
) -> dict[str, dict[str, Unit]]:
    """
    Generate unit dictionaries.

    Units for:

    mass, length, time, current, and temperature

    ..deprecated:: 3.1.0
        Everything is read directly out of the snapshots now.

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

    baryon = {
        "element_abundance": None,
        "maximal_temperature": temperature,
        "maximal_temperature_scale_factor": None,
        "maximal_temperature_time": time,
        "iron_mass_frac_from_sn1a": None,
        "metal_mass_frac_from_agb": None,
        "metal_mass_frac_from_snii": None,
        "metal_mass_frac_from_sn1a": None,
        "metallicity": None,
        "smoothed_element_abundance": None,
        "smoothed_iron_mass_frac_from_sn1a": None,
        "smoothed_metallicity": None,
        "total_mass_from_agb": mass,
        "total_mass_from_snii": mass,
    }

    gas = {
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
        **shared,
        **baryon,
    }

    dark_matter = {**shared}

    boundary = {**shared}

    sinks = {**shared}

    stars = {
        "birth_density": mass / (length**3),
        "birth_time": time,
        "initial_masses": mass,
        "smoothing_length": length,
        **shared,
        **baryon,
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

    ..deprecated:: 3.1.0
        Everything is read directly out of the snapshots now.
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
