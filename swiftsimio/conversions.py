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
        m_nu = None

        try:
            Tcmb0 = cosmo["T_CMB_0"][0]
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
            m_nu = cosmo["M_nu_eV"][0]
        except (IndexError, KeyError, AttributeError):
            m_nu = 0.0

        return w0waCDM(
            H0=H0.to_astropy(),
            Om0=Omega_m,
            Ode0=Omega_lambda,
            w0=w_0,
            wa=w_a,
            Tcmb0=Tcmb0,
            Ob0=Omega_b,
            Neff=Neff,
            m_nu=m_nu,
        )


else:

    def swift_cosmology_to_astropy(cosmo: dict, units) -> dict:
        return cosmo
