"""
A list of required fields for writing a SWIFT dataset, for each
particle type in turn.
"""

shared = {
    "coordinates": "Coordinates",
    "particle_ids": "ParticleIDs",
    "velocities": "Velocities",
    "masses": "Masses",
}

gas = {
    "smoothing_length": "SmoothingLength",
    "internal_energy": "InternalEnergy",
    **shared,
}

dark_matter = {**shared}

boundary = {**shared}

sinks = {**shared}

stars = {**shared, "smoothing_length": "SmoothingLength"}

black_holes = {**shared}

neutrinos = {**shared}
