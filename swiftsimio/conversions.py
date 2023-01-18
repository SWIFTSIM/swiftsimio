"""
Includes conversions between SWIFT internal values and
``astropy`` ones for convenience.
"""

from swiftsimio.optional_packages import ASTROPY_AVAILABLE
import unyt

if ASTROPY_AVAILABLE:
    from astropy.cosmology import w0waCDM
    from astropy.cosmology.core import Cosmology
    import astropy.version
    import astropy.constants as const
    import astropy.units as astropy_units
    import numpy as np

    def swift_neutrinos_to_astropy(N_eff, N_ur, M_nu_eV, deg_nu):
        """
        Parameters
        ----------

        N_eff: float
            Fractional number of effective massless neutrinos at high redshift

        N_ur: float
            Fractional number of massless neutrino species

        M_nu_eV: array of floats
            Masses in eV of massive species only, up to degeneracy

        deg_nu: array of floats
            Fractional degeneracies of the massive neutrino species

        Returns
        -------

        ap_m_nu
            Array of neutrino masses in eV, replicated according to degeneracy,
            including massless species, as desired by astropy
        """
        if np.isscalar(deg_nu):
            deg_nu = np.array([deg_nu])
        if np.isscalar(M_nu_eV):
            M_nu_eV = np.array([M_nu_eV])
        if not (deg_nu == deg_nu.astype(int)).all():
            raise AttributeError(
                "SWIFTsimIO uses astropy, which cannot handle this cosmological model."
            )
        if not int(N_eff) == deg_nu.astype(int).sum() + int(N_ur):
            raise AttributeError(
                "SWIFTsimIO uses astropy, which cannot handle this cosmological model."
            )
        ap_m_nu = [[m] * int(d) for m, d in zip(M_nu_eV, deg_nu)]  # replicate
        ap_m_nu = sum(ap_m_nu, []) + [0.0] * int(N_ur)  # flatten + add massless
        ap_m_nu = np.array(ap_m_nu) * astropy_units.eV
        return ap_m_nu

    def swift_cosmology_to_astropy(cosmo: dict, units) -> Cosmology:
        """
        Parameters
        ----------

        cosmo: dict
            Cosmology dictionary ready straight out of the SWIFT snapshot.

        units: SWIFTUnits
            The SWIFT Units instance associated with this snapshot.

        Returns
        -------

        Cosmology
            An instance of ``astropy.cosmology.Cosmology`` filled with the
            correct parameters.
        """

        H0 = unyt.unyt_quantity(cosmo["H0 [internal units]"][0], units=1.0 / units.time)

        Omega_b = cosmo["Omega_b"][0]
        Omega_lambda = cosmo["Omega_lambda"][0]
        Omega_r = cosmo["Omega_r"][0]
        Omega_m = cosmo["Omega_m"][0]
        w_0 = cosmo["w_0"][0]
        w_a = cosmo["w_a"][0]

        # For backwards compatibility with previous cosmology constructs
        # in snapshots
        Tcmb0 = None
        Neff = None
        N_ur = None
        M_nu_eV = None
        deg_nu = None

        try:
            Tcmb0 = cosmo["T_CMB_0 [K]"][0]
        except (IndexError, KeyError, AttributeError):
            # expressions taken directly from astropy, since they do no longer
            # allow access to these attributes (since version 5.1+)
            critdens_const = (3.0 / (8.0 * np.pi * const.G)).cgs.value
            a_B_c2 = (4.0 * const.sigma_sb / const.c ** 3).cgs.value

            # SWIFT provides Omega_r, but we need a consistent Tcmb0 for astropy.
            # This is an exact inversion of the procedure performed in astropy.
            critical_density_0 = astropy_units.Quantity(
                critdens_const * H0.to("1/s").value ** 2,
                astropy_units.g / astropy_units.cm ** 3,
            )

            Tcmb0 = (Omega_r * critical_density_0.value / a_B_c2) ** (1.0 / 4.0)

        try:
            Neff = cosmo["N_eff"][0]
        except (IndexError, KeyError, AttributeError):
            Neff = 3.04  # Astropy default

        try:
            M_nu_eV = cosmo["M_nu_eV"]
        except (IndexError, KeyError, AttributeError):
            M_nu_eV = 0.0

        try:
            deg_nu = cosmo["deg_nu"]
        except (IndexError, KeyError, AttributeError):
            deg_nu = 0.0

        try:
            N_ur = cosmo["N_ur"]
        except (IndexError, KeyError, AttributeError):
            N_ur = 3.04  # Astropy default

        ap_m_nu = swift_neutrinos_to_astropy(Neff, N_ur, M_nu_eV, deg_nu)

        return w0waCDM(
            H0=H0.to_astropy(),
            Om0=Omega_m,
            Ode0=Omega_lambda,
            w0=w_0,
            wa=w_a,
            Tcmb0=Tcmb0,
            Ob0=Omega_b,
            Neff=Neff,
            m_nu=ap_m_nu,
        )


else:

    def swift_cosmology_to_astropy(cosmo: dict, units) -> dict:
        return cosmo
