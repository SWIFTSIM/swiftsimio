"""
Contains the information for the units that determine the
particle fields. This must be provided, if not swiftsimio
will crash (as it should, you can't just be going around
having quantities without units...)

Unfortunately there must be a generator function used here
because we don't know the units ahead of time.
"""

from unyt import g, cm, s, statA, K


def generate_units(m, l, t, I, T):
    """
    Generates the unit dictionaries with the:

    mass, length, time, current, and temperature

    units respectively.
    """

    shared = {
        "coordinates": l,
        "masses": m,
        "particle_ids": None,
        "velocities": l / t
    }

    gas = {
        "density": m / (l**3),
        "entropy": m * l**2 / (t**2 * T),
        "internal_energy": m * l**2 / (t**2),
        "smoothing_length": l,
        "pressure": m / (l * t**2),
        **shared
    }

    dark_matter = {**shared}

    stars = {**shared}

    black_holes = {**shared}

    return {
        "gas": gas,
        "dark_matter": dark_matter,
        "stars": stars,
        "black_holes": black_holes
    }