"""
Contains the particle fields for the various particle types.
"""

shared = {
    "Coordinates": "coordinates",
    "Masses": "masses",
    "ParticleIDs": "particle_ids",
    "Velocities": "velocities",
}

gas = {
    "Density": "density",
    "Entropy": "entropy",
    "InternalEnergy": "internal_energy",
    "SmoothingLength": "smoothing_length",
    "Pressure": "pressure",
    **shared,
}

dark_matter = {**shared}

stars = {**shared}

black_holes = {**shared}
