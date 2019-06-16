"""
Runs tests on some fetched data from the web.

This will ensure that all particle fields are populated correctly, and that they can
be read in.
"""

from tests.helper import requires
from swiftsimio import load, mask

from unyt import unyt_array as array
from numpy import logical_and


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


@requires("comsological_volume.hdf5")
def test_reading_select_region_metadata(filename):
    """
    Tests reading select regions of the volume.
    """

    full_data = load(filename)

    # Mask off the lower bottom corner of the volume.
    mask_region = mask(filename, spatial_only=True)

    restrict = array(
        [
            [0.0, 0.0, 0.0] * full_data.metadata.boxsize.units,
            full_data.metadata.boxsize * 0.5,
        ]
    ).T

    mask_region.constrain_spatial(restrict=restrict)

    selected_data = load(filename, mask=mask_region)

    selected_coordinates = selected_data.gas.coordinates

    # Now need to repeat teh selection by hand:
    subset_mask = logical_and.reduce(
        [
            logical_and(x > y_lower, x < y_upper)
            for x, (y_lower, y_upper) in zip(full_data.gas.coordinates.T, restrict)
        ]
    )

    hand_selected_coordinates = full_data.gas.coordinates[subset_mask]

    assert (hand_selected_coordinates == selected_coordinates).all()

    return


@requires("cosmological_volume.hdf5")
def test_reading_select_region_metadata_not_spatial_only(filename):
    """
    The same as test_reading_select_region_metadata but for spatial_only=False.
    """

    full_data = load(filename)

    # Mask off the lower bottom corner of the volume.
    mask_region = mask(filename, spatial_only=False)

    restrict = array(
        [
            [0.0, 0.0, 0.0] * full_data.metadata.boxsize.units,
            full_data.metadata.boxsize * 0.5,
        ]
    ).T

    mask_region.constrain_spatial(restrict=restrict)

    selected_data = load(filename, mask=mask_region)

    selected_coordinates = selected_data.gas.coordinates

    # Now need to repeat teh selection by hand:
    subset_mask = logical_and.reduce(
        [
            logical_and(x > y_lower, x < y_upper)
            for x, (y_lower, y_upper) in zip(full_data.gas.coordinates.T, restrict)
        ]
    )

    hand_selected_coordinates = full_data.gas.coordinates[subset_mask]

    assert (hand_selected_coordinates == selected_coordinates).all()

    return
