"""
A list of required fields for writing a SWIFT dataset, for each
particle type in turn.
"""

_shared = {
    "coordinates": "Coordinates",
    "particle_ids": "ParticleIDs",
    "velocities": "Velocities",
    "masses": "Masses",
}

gas = {
    "smoothing_length": "SmoothingLength",
    "internal_energy": "InternalEnergy",
    **_shared,
}

dark_matter = {**_shared}

boundary = {**_shared}

sinks = {**_shared}

stars = {**_shared, "smoothing_length": "SmoothingLength"}

black_holes = {**_shared}

neutrinos = {**_shared}
