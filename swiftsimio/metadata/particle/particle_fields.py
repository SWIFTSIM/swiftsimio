"""
Contains the particle fields for the various particle types.
"""

shared = {
    "Coordinates": "coordinates",
    "Masses": "masses",
    "ParticleIDs": "particle_ids",
    "Velocities": "velocities",
    "Potential": "potential",
}

baryon = {
    "ElementAbundance": "element_abundance",
    "Maximal Temperature": "maximal_temperature",
    "Maximal Temperature scale-factor": "maximal_temperature_scale_factor",
    "IronMassFracFromSNIa": "iron_mass_frac_from_sn1a",
    "MetalMassFracFromAGB": "metal_mass_frac_from_agb",
    "MetalMassFracFromSNII": "metal_mass_frac_from_snii",
    "MetalMassFracFromSNIa": "metal_mass_frac_from_sn1a",
    "Metallicity": "metallicity",
    "SmoothedElementAbundance": "smoothed_element_abundance",
    "SmoothedIronMassFracFromSNIa": "smoothed_iron_mass_frac_from_sn1a",
    "SmoothedMetallicity": "smoothed_metallicity",
    "TotalMassFromAGB": "total_mass_from_agb",
    "TotalMassFromSNII": "total_mass_from_snii",
}

gas = {
    "Density": "density",
    "Entropy": "entropy",
    "InternalEnergy": "internal_energy",
    "SmoothingLength": "smoothing_length",
    "Pressure": "pressure",
    "Diffusion": "diffusion",
    "SFR": "sfr",
    "Temperature": "temperature",
    "Viscosity": "viscosity",
    "sSFR": "specific_sfr",
    **shared,
    **baryon,
}

dark_matter = {**shared}

stars = {
    "BirthDensity": "birth_density",
    "Birth_time": "birth_time",
    "Initial_Masses": "initial_masses",
    "NewStarFlag": "new_star_flag",
    **shared,
    **baryon,
}

black_holes = {**shared}
