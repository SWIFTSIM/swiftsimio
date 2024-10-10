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

stars = { 
    "smoothing_length": "SmoothingLength",
    "stellar_particle_type": "StellarParticleType",
    **shared,
}

black_holes = {**shared}

neutrinos = {**shared}
