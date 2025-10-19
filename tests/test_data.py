"""
Runs tests on some fetched data from the web.

This will ensure that all particle fields are populated correctly, and that they can
be read in.
"""

import numpy as np
from swiftsimio import load
from .helper import _mask_without_warning as mask
from os import remove

import h5py

from unyt import K, Msun
from numpy import logical_and, isclose, float64
from numpy import array as numpy_array

from swiftsimio.objects import cosmo_array
from tests.helper import create_n_particle_dataset


def test_cosmology_metadata(cosmological_volume):
    """
    Tests to see if we get the unpacked cosmology metadata correct.
    """
    data = load(cosmological_volume)

    assert data.metadata.a == data.metadata.scale_factor
    assert np.isclose(data.metadata.a, 1.0 / (1.0 + data.metadata.redshift), atol=1e-8)

    return


def test_time_metadata(cosmological_volume):
    """
    This tests the time metadata and also tests the ability to include two items at once
    from the same header attribute.
    """
    data = load(cosmological_volume)

    assert data.metadata.z == data.metadata.redshift

    assert data.metadata.t == data.metadata.time

    return


def test_temperature_units(cosmological_volume):
    """
    This tests checks if we correctly read in temperature units. Based
    on a past bug, to make sure we never break this again.
    """
    data = load(cosmological_volume)
    data.gas.temperatures.convert_to_units(K)
    return


def test_initial_mass_table(cosmological_volume):
    """
    This tests checks if we correctly read in the initial mass table. Based
    on a past bug, to make sure we never break this again.
    """
    data = load(cosmological_volume)
    data.metadata.initial_mass_table.gas.convert_to_units(Msun)

    return


def test_units(cosmological_volume):
    """
    Tests that these fields have the same units within SWIFTsimIO as they
    do in the SWIFT code itself.
    """
    data = load(cosmological_volume)

    shared = ["coordinates", "masses", "particle_ids", "velocities"]

    baryon = [
        "maximal_temperatures",
        "maximal_temperature_scale_factors",
        "iron_mass_fractions_from_snia",
        "metal_mass_fractions_from_agb",
        "metal_mass_fractions_from_snii",
        "metal_mass_fractions_from_snia",
        "smoothed_iron_mass_fractions_from_snia",
        "smoothed_metal_mass_fractions",
    ]

    to_test = {
        "gas": [
            "densities",
            "entropies",
            "internal_energies",
            "smoothing_lengths",
            "pressures",
            "temperatures",
        ]
        + baryon
        + shared,
        "dark_matter": shared,
    }

    for ptype, properties in to_test.items():
        field = getattr(data, ptype)

        # Now need to extract the particle paths in the original hdf5 file
        # for comparison...
        paths = numpy_array(field.group_metadata.field_paths)
        names = numpy_array(field.group_metadata.field_names)

        for property in properties:
            # Read the 0th element, and compare in CGS units.
            # We need to use doubles here as sometimes we can overflow!
            our_units = getattr(field, property).astype(float64)[0]

            our_units.convert_to_cgs()

            # Find the path in the HDF5 for our linked dataset
            path = paths[names == property][0]

            with h5py.File(cosmological_volume, "r") as handle:
                swift_units = handle[path].attrs[
                    "Conversion factor to CGS (not including cosmological corrections)"
                ][0]
                swift_value = swift_units * handle[path][0]

            assert isclose(swift_value, our_units.value, 5e-5).all()

    # If we didn't crash out, we gucci.
    return


def test_cell_metadata_is_valid(cosmological_volume):
    """
    Test that the metadata does what we think it does!

    I.e. that it sets the particles contained in a top-level cell.
    """
    mask_region = mask(cosmological_volume)
    # Because we sort by offset if we are using the metadata we
    # must re-order the data to be in the correct order
    mask_region.constrain_spatial(
        cosmo_array(
            [np.zeros_like(mask_region.metadata.boxsize), mask_region.metadata.boxsize]
        ).T
    )
    data = load(cosmological_volume, mask=mask_region)

    cell_size = mask_region.cell_size.to(data.gas.coordinates.units)
    boxsize = mask_region.metadata.boxsize[0].to(data.gas.coordinates.units)
    offsets = mask_region.offsets["gas"]
    counts = mask_region.counts["gas"]

    start_offset = offsets
    stop_offset = offsets + counts

    for center, start, stop in zip(
        mask_region.centers.to(data.gas.coordinates.units), start_offset, stop_offset
    ):
        for dimension in range(0, 3):
            lower = (center - 0.5 * cell_size)[dimension]
            upper = (center + 0.5 * cell_size)[dimension]

            max = data.gas.coordinates[start:stop, dimension].max()
            min = data.gas.coordinates[start:stop, dimension].min()

            # Ignore things close to the boxsize
            if min < 0.05 * boxsize or max > 0.95 * boxsize:
                continue

            # Give it a little wiggle room.
            assert max <= upper * 1.05
            assert min > lower * 0.95


def test_dithered_cell_metadata_is_valid(cosmological_volume_dithered):
    """
    Test that the metadata does what we think it does, in the
    dithered case.

    I.e. that it sets the particles contained in a top-level cell.
    """
    mask_region = mask(cosmological_volume_dithered)
    # Because we sort by offset if we are using the metadata we
    # must re-order the data to be in the correct order
    mask_region.constrain_spatial(
        cosmo_array(
            [np.zeros_like(mask_region.metadata.boxsize), mask_region.metadata.boxsize]
        ).T
    )
    data = load(cosmological_volume_dithered, mask=mask_region)

    cell_size = mask_region.cell_size.to(data.dark_matter.coordinates.units)
    boxsize = mask_region.metadata.boxsize[0].to(data.dark_matter.coordinates.units)
    offsets = mask_region.offsets["dark_matter"]
    counts = mask_region.counts["dark_matter"]

    start_offset = offsets
    stop_offset = offsets + counts

    for center, start, stop in zip(
        mask_region.centers.to(data.dark_matter.coordinates.units),
        start_offset,
        stop_offset,
    ):
        for dimension in range(0, 3):
            lower = (center - 0.5 * cell_size)[dimension]
            upper = (center + 0.5 * cell_size)[dimension]

            max = data.dark_matter.coordinates[start:stop, dimension].max()
            min = data.dark_matter.coordinates[start:stop, dimension].min()

            # Ignore things close to the boxsize
            if min < 0.05 * boxsize or max > 0.95 * boxsize:
                continue

            # Give it a little wiggle room
            assert max <= upper * 1.05
            assert min > lower * 0.95


def test_reading_select_region_metadata(cosmological_volume):
    """
    Tests reading select regions of the volume.
    """
    full_data = load(cosmological_volume)

    # Mask off the centre of the volume.
    mask_region = mask(cosmological_volume, spatial_only=True)

    restrict = cosmo_array(
        [full_data.metadata.boxsize * 0.2, full_data.metadata.boxsize * 0.8]
    ).T

    mask_region.constrain_spatial(restrict=restrict)

    selected_data = load(cosmological_volume, mask=mask_region)

    selected_coordinates = selected_data.gas.coordinates

    # Now need to repeat teh selection by hand:

    subset_mask = logical_and.reduce(
        [
            logical_and(x > y_lower, x < y_upper)
            for x, (y_lower, y_upper) in zip(full_data.gas.coordinates.T, restrict)
        ]
    )

    # We also need to repeat for the thing we just selected; the cells only give
    # us an _approximate_ selection!
    selected_subset_mask = logical_and.reduce(
        [
            logical_and(x > y_lower, x < y_upper)
            for x, (y_lower, y_upper) in zip(selected_data.gas.coordinates.T, restrict)
        ]
    )

    hand_selected_coordinates = full_data.gas.coordinates[subset_mask]

    assert (
        hand_selected_coordinates.value
        == selected_coordinates[selected_subset_mask].value
    ).all()
    return


def test_reading_select_region_metadata_not_spatial_only(cosmological_volume):
    """
    The same as test_reading_select_region_metadata but for spatial_only=False.
    """
    full_data = load(cosmological_volume)

    # Mask off the centre of the volume.
    mask_region = mask(cosmological_volume, spatial_only=False)

    restrict = cosmo_array(
        [full_data.metadata.boxsize * 0.26, full_data.metadata.boxsize * 0.74]
    ).T

    mask_region.constrain_spatial(restrict=restrict)

    selected_data = load(cosmological_volume, mask=mask_region)

    selected_coordinates = selected_data.gas.coordinates

    # Now need to repeat the selection by hand:
    subset_mask = logical_and.reduce(
        [
            logical_and(x > y_lower, x < y_upper)
            for x, (y_lower, y_upper) in zip(full_data.gas.coordinates.T, restrict)
        ]
    )

    # We also need to repeat for the thing we just selected; the cells only give
    # us an _approximate_ selection!
    selected_subset_mask = logical_and.reduce(
        [
            logical_and(x > y_lower, x < y_upper)
            for x, (y_lower, y_upper) in zip(selected_data.gas.coordinates.T, restrict)
        ]
    )

    hand_selected_coordinates = full_data.gas.coordinates[subset_mask]

    assert (
        hand_selected_coordinates == selected_coordinates[selected_subset_mask]
    ).all()

    return


def test_reading_empty_dataset(cosmological_volume):
    """
    Test that we can read in a zero-particle dataset (e.g. Type4 or Type5 present in
    snapshot but no particles exist yet). Here we make a snapshot file with Type1
    present but empty.
    """
    # unmasked case
    output_filename = "zero_particle.hdf5"
    create_n_particle_dataset(cosmological_volume, output_filename, num_parts=0)
    data = load(output_filename)
    assert data.gas.masses.shape == (0,)
    assert data.gas.coordinates.shape == (0, 3)
    try:
        assert data.gas.element_mass_fractions.hydrogen.shape == (0,)
    except AttributeError:
        # in the LegacyCosmologicalVolume case
        assert data.gas.element_mass_fractions.shape == (0, 9)

    # masked case
    m = mask(output_filename)
    m.constrain_spatial(
        cosmo_array([np.zeros_like(m.metadata.boxsize), 0.5 * m.metadata.boxsize]).T
    )
    masked_data = load(output_filename, mask=m)
    assert masked_data.gas.masses.shape == (0,)
    assert masked_data.gas.coordinates.shape == (0, 3)
    try:
        assert masked_data.gas.element_mass_fractions.hydrogen.shape == (0,)
    except AttributeError:
        # in the LegacyCosmologicalVolume case
        assert masked_data.gas.element_mass_fractions.shape == (0, 9)

    # clean up
    remove(output_filename)
