"""Deprecated: define metadata for the cosmology fields by hand."""

from sympy import Expr
from swiftsimio.objects import cosmo_factor, a

a_exponents = {"coordinates": 1, "internal_energies": -2}


def generate_cosmology(
    scale_factor: float, gamma: float
) -> dict[str, dict[str, cosmo_factor]]:
    """
    Generate the cosmology dictionaries for each particle field.

    Parameters
    ----------
    scale_factor : float
        The scale factor.

    gamma : float
        The gas adiabatic index.

    Returns
    -------
    out : dict[str, dict[str, cosmo_factor]]
        Dictionary of ``cosmo_factor``s for particle fields.

    ..deprecated:: 3.1.0
        This information is now directly read out of the snapshots.
    """

    def cosmo_factory(a_dependence: Expr) -> cosmo_factor:
        """
        Generate a ``cosmo_factor``.

        Parameters
        ----------
        a_dependence : Expr
            The scale factor dependence desired.

        Returns
        -------
        out : cosmo_factor
            One of our ``cosmo_factor`` objects.
        """
        return cosmo_factor(a_dependence, scale_factor)

    # TODO
    no_cosmology = cosmo_factory(a / a)
    UNKNOWN = no_cosmology

    shared = {
        "coordinates": cosmo_factory(a),
        "masses": no_cosmology,
        "particle_ids": no_cosmology,
        "velocities": UNKNOWN,
        "potential": UNKNOWN,
    }

    baryon = {
        "element_abundance": no_cosmology,
        "maximal_temperature": no_cosmology,
        "maximal_temperature_scale_factor": no_cosmology,
        "maximal_temperature_time": no_cosmology,
        "iron_mass_frac_from_sn1a": no_cosmology,
        "metal_mass_frac_from_agb": no_cosmology,
        "metal_mass_frac_from_snii": no_cosmology,
        "metal_mass_frac_from_sn1a": no_cosmology,
        "metallicity": no_cosmology,
        "smoothed_element_abundance": no_cosmology,
        "smoothed_iron_mass_frac_from_sn1a": no_cosmology,
        "smoothed_metallicity": no_cosmology,
        "total_mass_from_agb": no_cosmology,
        "total_mass_from_snii": no_cosmology,
    }

    gas = {
        "density": cosmo_factory(a ** (-3)),
        "entropy": no_cosmology,
        "internal_energy": cosmo_factory(a ** (-3.0 * (gamma - 1))),
        "smoothing_length": cosmo_factory(a),
        "pressure": cosmo_factory(a ** (-3.0 * gamma)),
        "diffusion": no_cosmology,
        "sfr": no_cosmology,
        "temperature": no_cosmology,
        "specific_sfr": no_cosmology,
        "material_id": no_cosmology,
        "viscosity": no_cosmology,
        "radiated_energy": no_cosmology,
        **shared,
        **baryon,
    }

    dark_matter = {**shared}

    boundary = {**shared}

    sinks = {**shared}

    stars = {
        "birth_density": gas["density"],
        "birth_time": no_cosmology,
        "initial_masses": no_cosmology,
        "smoothing_length": cosmo_factory(a),
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
