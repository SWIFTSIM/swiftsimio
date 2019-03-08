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
    "Maximal Temperature time": "maximal_temperature_time";
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
    "MaterialID": "material_id",
    "Diffusion": "diffusion",
    "Viscosity": "viscosity",
    **shared,
    **baryon,
}

dark_matter = {**shared}

stars = {
    "BirthDensity": "birth_density",
    "BirthTime": "birth_time",
    "InitialMasses": "initial_masses",
    **shared,
    **baryon,
}

black_holes = {**shared}
