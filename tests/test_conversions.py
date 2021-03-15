"""
Tests conversions from SWIFT internals to astropy.
"""

from swiftsimio.conversions import swift_cosmology_to_astropy
from tests.helper import requires
from swiftsimio import load
from numpy import isclose


@requires("cosmological_volume.hdf5")
def test_basic_tcmb(filename):
    """
    Tests we can recover omega_gamma = 0.0 and tcmb0 in the usual case.
    """

    data = load(filename)

    try:
        assert (
            data.metadata.cosmology._Ogamma0
            == data.metadata.cosmology_raw["Omega_r"][0]
        )
    except AttributeError:
        # Broken astropy install
        pass


@requires("cosmological_volume.hdf5")
def test_nonzero_tcmb(filename):
    """
    Tests we can recover omega_gamma = 0.0 and tcmb0 in the usual case.
    """

    data = load(filename)
    units = data.metadata.units

    cosmo = data.metadata.cosmology_raw

    cosmo["Omega_r"] = [0.1]

    output_cosmology = swift_cosmology_to_astropy(cosmo=cosmo, units=units)

    assert isclose(output_cosmology._Ogamma0, 0.1)
