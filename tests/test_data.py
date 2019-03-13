"""
Runs tests on some fetched data from the web.

This will ensure that all particle fields are populated correctly, and that they can
be read in.
"""

from tests.helper import requires
from swiftsimio import load


@requires("cosmological_volume.hdf5")
def test_cosmology_metadata(filename):
    """
    Tests to see if we get the unpacked cosmology metadata correct.
    """

    data = load(filename)

    assert data.metadata.a == data.metadata.scale_factor

    assert data.metadata.a == 1.0 / (1.0 + data.metadata.redshift)

    return


@requires("cosmological_volume.hdf5")
def test_time_metadata(filename):
    """
    This tests the time metadata and also tests the ability to include two items at once from
    the same header attribute.
    """

    data = load(filename)

    assert data.metadata.z == data.metadata.redshift

    assert data.metadata.t == data.metadata.time

    return


@requires("cosmological_volume.hdf5")
def test_fields_present(filename):
    """
    Tests that all EAGLE-related fields are present in the cosmological
    volume example and that they match with what's in swiftsimio.
    """

    data = load(filename)

    shared = ["coordinates", "masses", "particle_ids", "velocities"]

    baryon = [
        "element_abundance",
        "maximal_temperature",
        "maximal_temperature_scale_factor",
        "iron_mass_frac_from_sn1a",
        "metal_mass_frac_from_agb",
        "metal_mass_frac_from_snii",
        "metal_mass_frac_from_sn1a",
        "metallicity",
        "smoothed_element_abundance",
        "smoothed_iron_mass_frac_from_sn1a",
        "smoothed_metallicity",
        "total_mass_from_agb",
        "total_mass_from_snii",
    ]

    to_test = {
        "gas": [
            "density",
            "entropy",
            "internal_energy",
            "smoothing_length",
            "pressure",
            "diffusion",
            "sfr",
            "temperature",
            "viscosity",
            "diffusion",
            "viscosity",
        ]
        + baryon
        + shared,
        "dark_matter": shared,
    }

    for ptype, properties in to_test.items():
        field = getattr(data, ptype)
        for property in properties:
            _ = getattr(field, property)

    # If we didn't crash out, we gucci.
    return
