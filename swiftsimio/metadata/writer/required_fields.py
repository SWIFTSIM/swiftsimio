"""A list of required fields for writing a SWIFT dataset, for each particle type."""

import unyt.dimensions as dim

_shared = {
    "coordinates": {"handle": "Coordinates", "dimensions": dim.length},
    "particle_ids": {"handle": "ParticleIDs", "dimensions": dim.dimensionless},
    "velocities": {"handle": "Velocities", "dimensions": dim.velocity},
    "masses": {"handle": "Masses", "dimensions": dim.mass},
}

gas = {
    "smoothing_length": {"handle": "SmoothingLength", "dimensions": dim.length},
    "internal_energy": {
        "handle": "InternalEnergy",
        "dimensions": dim.length**2 / dim.time**2,
    },
    **_shared,
}

dark_matter = {**_shared}

boundary = {**_shared}

sinks = {**_shared}

stars = {
    **_shared,
    "smoothing_length": {"handle": "SmoothingLength", "dimensions": dim.length},
}

black_holes = {**_shared}

neutrinos = {**_shared}
