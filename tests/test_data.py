"""
Runs tests on some fetched data from the web.

This will ensure that all particle fields are populated correctly, and that they can
be read in.
"""

import pytest

from tests.helper import requires
from swiftsimio import load, mask

import h5py

from unyt import K, Msun
from numpy import logical_and, isclose, float64
from numpy import array as numpy_array

from swiftsimio.objects import cosmo_array, cosmo_factor, a


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
def test_temperature_units(filename):
    """
    This tests checks if we correctly read in temperature units. Based
    on a past bug, to make sure we never break this again.
    """

    data = load(filename)
    data.gas.temperatures.convert_to_units(K)

    return

@requires("cosmological_volume.hdf5")
def test_initial_mass_table(filename):
    """
    This tests checks if we correctly read in the initial mass table. Based
    on a past bug, to make sure we never break this again.
    """

    data = load(filename)
    data.metadata.initial_mass_table.gas.convert_to_units(Msun)

    return


@requires("cosmological_volume.hdf5")
def test_units(filename):
    """
    Tests that these fields have the same units within SWIFTsimIO as they
    do in the SWIFT code itself.
    """

    data = load(filename)

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

            with h5py.File(filename, "r") as handle:
                swift_units = handle[path].attrs[
                    "Conversion factor to CGS (not including cosmological corrections)"
                ][0]
                swift_value = swift_units * handle[path][0]

            assert isclose(swift_value, our_units.value, 5e-5).all()

    # If we didn't crash out, we gucci.
    return


@requires("cosmological_volume.hdf5")
def test_cell_metadata_is_valid(filename):
    """
    Test that the metadata does what we think it does!

    I.e. that it sets the particles contained in a top-level cell.
    """

    mask_region = mask(filename)
    # Because we sort by offset if we are using the metadata we
    # must re-order the data to be in the correct order
    mask_region.constrain_spatial([[0 * b, b] for b in mask_region.metadata.boxsize])
    data = load(filename, mask=mask_region)

    cell_size = mask_region.cell_size.to(data.gas.coordinates.units)
    boxsize = mask_region.metadata.boxsize[0].to(data.gas.coordinates.units)
    offsets = mask_region.offsets["gas"]
    counts = mask_region.counts["gas"]

    # can be removed when issue #128 resolved:
    boxsize = cosmo_array(
        boxsize,
        comoving=True,
        cosmo_factor=cosmo_factor(a ** 1, mask_region.metadata.a),
    )

    start_offset = offsets
    stop_offset = offsets + counts

    print(mask_region.centers)
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
            # Mask_region provides unyt_array, not cosmo_array, anticipate warnings.
            with pytest.warns(RuntimeWarning, match="Mixing ufunc arguments"):
                assert max <= upper * 1.05
            with pytest.warns(RuntimeWarning, match="Mixing ufunc arguments"):
                assert min > lower * 0.95


@requires("cosmological_volume_dithered.hdf5")
def test_dithered_cell_metadata_is_valid(filename):
    """
    Test that the metadata does what we think it does, in the
    dithered case.

    I.e. that it sets the particles contained in a top-level cell.
    """

    mask_region = mask(filename)
    # Because we sort by offset if we are using the metadata we
    # must re-order the data to be in the correct order
    mask_region.constrain_spatial([[0 * b, b] for b in mask_region.metadata.boxsize])
    data = load(filename, mask=mask_region)

    cell_size = mask_region.cell_size.to(data.dark_matter.coordinates.units)
    boxsize = mask_region.metadata.boxsize[0].to(data.dark_matter.coordinates.units)
    # can be removed when issue #128 resolved:
    boxsize = cosmo_array(
        boxsize,
        comoving=True,
        cosmo_factor=cosmo_factor(a ** 1, mask_region.metadata.a),
    )
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
            # Mask_region provides unyt_array, not cosmo_array, anticipate warnings.
            with pytest.warns(RuntimeWarning, match="Mixing ufunc arguments"):
                assert max <= upper * 1.05
            # Mask_region provides unyt_array, not cosmo_array, anticipate warnings.
            with pytest.warns(RuntimeWarning, match="Mixing ufunc arguments"):
                assert min > lower * 0.95


@requires("cosmological_volume.hdf5")
def test_reading_select_region_metadata(filename):
    """
    Tests reading select regions of the volume.
    """

    full_data = load(filename)

    # Mask off the centre of the volume.
    mask_region = mask(filename, spatial_only=True)

    # can be removed when issue #128 resolved:
    boxsize = cosmo_array(
        full_data.metadata.boxsize,
        comoving=True,
        cosmo_factor=cosmo_factor(a ** 1, full_data.metadata.a),
    )
    restrict = cosmo_array([boxsize * 0.2, boxsize * 0.8]).T

    mask_region.constrain_spatial(restrict=restrict)

    selected_data = load(filename, mask=mask_region)

    selected_coordinates = selected_data.gas.coordinates

    # Now need to repeat teh selection by hand:
    # Iterating a cosmo_array gives unyt_quantities, anticipate the warning for comparing to cosmo_array.
    with pytest.warns(RuntimeWarning, match="Mixing ufunc arguments"):
        subset_mask = logical_and.reduce(
            [
                logical_and(x > y_lower, x < y_upper)
                for x, (y_lower, y_upper) in zip(full_data.gas.coordinates.T, restrict)
            ]
        )

    # We also need to repeat for the thing we just selected; the cells only give
    # us an _approximate_ selection!
    # Iterating a cosmo_array gives unyt_quantities, anticipate the warning for comparing to cosmo_array.
    with pytest.warns(RuntimeWarning, match="Mixing ufunc arguments"):
        selected_subset_mask = logical_and.reduce(
            [
                logical_and(x > y_lower, x < y_upper)
                for x, (y_lower, y_upper) in zip(
                    selected_data.gas.coordinates.T, restrict
                )
            ]
        )

    hand_selected_coordinates = full_data.gas.coordinates[subset_mask]

    assert (
        hand_selected_coordinates.value
        == selected_coordinates[selected_subset_mask].value
    ).all()
    return


@requires("cosmological_volume.hdf5")
def test_reading_select_region_metadata_not_spatial_only(filename):
    """
    The same as test_reading_select_region_metadata but for spatial_only=False.
    """

    full_data = load(filename)

    # Mask off the centre of the volume.
    mask_region = mask(filename, spatial_only=False)

    # can be removed when issue #128 resolved:
    boxsize = cosmo_array(
        full_data.metadata.boxsize,
        comoving=True,
        cosmo_factor=cosmo_factor(a ** 1, full_data.metadata.a),
    )
    restrict = cosmo_array([boxsize * 0.26, boxsize * 0.74]).T

    mask_region.constrain_spatial(restrict=restrict)

    selected_data = load(filename, mask=mask_region)

    selected_coordinates = selected_data.gas.coordinates

    # Now need to repeat the selection by hand:
    # Iterating a cosmo_array gives unyt_quantities, anticipate the warning for comparing to cosmo_array.
    with pytest.warns(RuntimeWarning, match="Mixing ufunc arguments"):
        subset_mask = logical_and.reduce(
            [
                logical_and(x > y_lower, x < y_upper)
                for x, (y_lower, y_upper) in zip(full_data.gas.coordinates.T, restrict)
            ]
        )

    # We also need to repeat for the thing we just selected; the cells only give
    # us an _approximate_ selection!
    # Iterating a cosmo_array gives unyt_quantities, anticipate the warning for comparing to cosmo_array.
    with pytest.warns(RuntimeWarning, match="Mixing ufunc arguments"):
        selected_subset_mask = logical_and.reduce(
            [
                logical_and(x > y_lower, x < y_upper)
                for x, (y_lower, y_upper) in zip(
                    selected_data.gas.coordinates.T, restrict
                )
            ]
        )

    hand_selected_coordinates = full_data.gas.coordinates[subset_mask]

    assert (
        hand_selected_coordinates == selected_coordinates[selected_subset_mask]
    ).all()

    return
