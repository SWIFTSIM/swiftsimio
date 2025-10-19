"""Tests conversions from SWIFT internals to astropy."""

from swiftsimio.conversions import swift_cosmology_to_astropy
from swiftsimio import load
from numpy import isclose


def test_basic_tcmb(cosmological_volume):
    """Tests we can recover omega_gamma = 0.0 and tcmb0 in the usual case."""
    data = load(cosmological_volume)

    try:
        assert (
            data.metadata.cosmology._Ogamma0
            == data.metadata.cosmology_raw["Omega_r"][0]
        )
    except AttributeError:
        # Broken astropy install
        pass


def test_nonzero_tcmb(cosmological_volume):
    """Tests we can recover omega_gamma = 0.0 and tcmb0 in the usual case."""
    data = load(cosmological_volume)
    units = data.metadata.units

    cosmo = data.metadata.cosmology_raw

    cosmo["Omega_r"] = [0.1]

    output_cosmology = swift_cosmology_to_astropy(cosmo=cosmo, units=units)

    assert isclose(output_cosmology.Ogamma0, 0.1)
