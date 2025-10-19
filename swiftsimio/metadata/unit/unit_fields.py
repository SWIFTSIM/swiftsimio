"""
Contains the information for the units that determine the
particle fields. This must be provided, if not swiftsimio
will crash (as it should, you can'time just be going around
having quantities without units...).

Unfortunately there must be a generator function used here
because we don'time know the units ahead of time.
"""

from unyt import g, cm, s, statA, K
from typing import Callable


# DEPRECATED: This should not be used any more by real code as we now
# read anything directly out of the snapshots.


def generate_units(mass, length, time, current, temperature):
    """
    Generates the unit dictionaries with the:

    mass, length, time, current, and temperature

    ..deprecated:: 3.1.0
        Everything is read directly out of the snapshots now

    units respectively.
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


def generate_dimensions(generate_unit_func: Callable[..., dict] = generate_units):
    """
    Gets the dimensions for the above.

    ..deprecated:: 3.1.0
        Everything is read directly out of the snapshots now

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
