"""
Contains the information for the units that determine the
particle fields. This must be provided, if not swiftsimio
will crash (as it should, you can't just be going around
having quantities without units...)

Unfortunately there must be a generator function used here
because we don't know the units ahead of time.
"""

from unyt import g, cm, s, statA, K
from typing import Callable


# DEPRECATED: This should not be used any more by real code as we now
# read anything directly out of the snapshots.


def generate_units(m, l, t, I, T):
    """
    Generates the unit dictionaries with the:

    mass, length, time, current, and temperature

    ..deprecated:: 3.1.0
        Everything is read directly out of the snapshots now

    units respectively.
    """

    shared = {
        "coordinates": l,
        "masses": m,
        "particle_ids": None,
        "velocities": l / t,
        "potential": l * l / (t * t),
    }

    baryon = {
        "element_abundance": None,
        "maximal_temperature": T,
        "maximal_temperature_scale_factor": None,
        "maximal_temperature_time": t,
        "iron_mass_frac_from_sn1a": None,
        "metal_mass_frac_from_agb": None,
        "metal_mass_frac_from_snii": None,
        "metal_mass_frac_from_sn1a": None,
        "metallicity": None,
        "smoothed_element_abundance": None,
        "smoothed_iron_mass_frac_from_sn1a": None,
        "smoothed_metallicity": None,
        "total_mass_from_agb": m,
        "total_mass_from_snii": m,
    }

    gas = {
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
        **shared,
        **baryon,
    }

    dark_matter = {**shared}

    boundary = {**shared}

    second_boundary = {**shared}

    stars = {
        "birth_density": m / (l ** 3),
        "birth_time": t,
        "initial_masses": m,
        "smoothing_length": l,
        **shared,
        **baryon,
    }

    black_holes = {**shared}

    return {
        "gas": gas,
        "dark_matter": dark_matter,
        "boundary": boundary,
        "second_boundary": second_boundary,
        "stars": stars,
        "black_holes": black_holes,
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
