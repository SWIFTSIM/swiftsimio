"""
Tests the masking using some test data.
"""

import h5py
import pytest
from swiftsimio import load, mask, cosmo_array, cosmo_quantity
import numpy as np
import unyt as u


def test_reading_select_region_spatial(cosmological_volume):
    """
    Tests reading select regions of the volume, comparing the masks attained with
    spatial_only = True and spatial_only = False.
    """

    full_data = load(cosmological_volume)

    # Mask off the lower bottom corner of the volume.
    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        mask_region = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region = mask(cosmological_volume, spatial_only=True)
    if has_cell_bbox:
        mask_region_nospatial = mask(cosmological_volume, spatial_only=False)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region_nospatial = mask(cosmological_volume, spatial_only=False)

    restrict = cosmo_array(
        [np.zeros_like(full_data.metadata.boxsize), full_data.metadata.boxsize * 0.5]
    ).T

    mask_region.constrain_spatial(restrict=restrict)
    mask_region_nospatial.constrain_spatial(restrict=restrict)

    selected_data = load(cosmological_volume, mask=mask_region)
    selected_data_nospatial = load(cosmological_volume, mask=mask_region_nospatial)

    selected_coordinates = selected_data.gas.coordinates
    selected_coordinates_nospatial = selected_data_nospatial.gas.coordinates

    assert (selected_coordinates_nospatial == selected_coordinates).all()

    return


def test_reading_select_region_half_box(cosmological_volume):
    """
    Tests reading the spatial region and sees if it lies within the region
    we told it to!

    Specifically, we test to see if all particles lie within half a boxsize.
    """

    # Mask off the lower bottom corner of the volume.
    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        mask_region = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region = mask(cosmological_volume, spatial_only=True)

    # the region can be padded by a cell if min & max particle positions are absent
    # in metadata
    pad_frac = (mask_region.cell_size / mask_region.metadata.boxsize).to_value(
        u.dimensionless
    )
    restrict = cosmo_array(
        [
            mask_region.metadata.boxsize * (pad_frac + 0.01),
            mask_region.metadata.boxsize * (0.5 - pad_frac - 0.01),
        ]
    ).T

    mask_region.constrain_spatial(restrict=restrict)

    selected_data = load(cosmological_volume, mask=mask_region)

    selected_coordinates = selected_data.gas.coordinates
    # Some of these particles will be outside because of the periodic BCs
    assert (
        (selected_coordinates / mask_region.metadata.boxsize).to_value(u.dimensionless)
        > 0.5 + pad_frac
    ).sum() < 25
    # make sure the test isn't trivially passed because we selected nothing:
    assert selected_coordinates.size > 0


def test_region_mask_not_modified(cosmological_volume):
    """
    Tests if a mask region is modified during the course of its use.

    Checks if https://github.com/SWIFTSIM/swiftsimio/issues/22 is broken.
    """

    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        this_mask = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            this_mask = mask(cosmological_volume, spatial_only=True)
    bs = this_mask.metadata.boxsize

    read = [[0 * b, 0.5 * b] for b in bs]
    read_constant = [[0 * b, 0.5 * b] for b in bs]

    this_mask._generate_cell_mask(read)

    assert read == read_constant


def test_region_mask_intersection(cosmological_volume):
    """
    Tests that the intersection of two spatial mask regions includes the same cells as two
    separate masks of the same two regions.
    """
    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        mask_1 = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_1 = mask(cosmological_volume, spatial_only=True)
    if has_cell_bbox:
        mask_2 = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_2 = mask(cosmological_volume, spatial_only=True)
    if has_cell_bbox:
        mask_intersect = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_intersect = mask(cosmological_volume, spatial_only=True)
    bs = mask_intersect.metadata.boxsize
    region_1 = [[0 * b, 0.1 * b] for b in bs]
    region_2 = [[0.6 * b, 0.7 * b] for b in bs]
    mask_1.constrain_spatial(region_1)
    mask_2.constrain_spatial(region_2)
    # the intersect=True flag is optional on the first call:
    mask_intersect.constrain_spatial(region_1, intersect=True)
    mask_intersect.constrain_spatial(region_2, intersect=True)
    for group_name in mask_1.metadata.present_group_names:
        assert (
            np.logical_or(mask_1.cell_mask[group_name], mask_2.cell_mask[group_name])
            == mask_intersect.cell_mask[group_name]
        ).all()


def test_mask_periodic_wrapping(cosmological_volume):
    """
    Check that a region that runs off the upper edge of the box gives the same
    mask as one that runs off the lower edge (they are chosen to be equivalent
    under periodic wrapping).
    """
    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        mask_region_upper = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region_upper = mask(cosmological_volume, spatial_only=True)
    if has_cell_bbox:
        mask_region_lower = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region_lower = mask(cosmological_volume, spatial_only=True)
    restrict_upper = cosmo_array(
        [
            mask_region_upper.metadata.boxsize * 0.8,
            mask_region_upper.metadata.boxsize * 1.2,
        ]
    ).T
    restrict_lower = cosmo_array(
        [
            mask_region_lower.metadata.boxsize * (-0.2),
            mask_region_lower.metadata.boxsize * 0.2,
        ]
    ).T

    mask_region_upper.constrain_spatial(restrict=restrict_upper)
    mask_region_lower.constrain_spatial(restrict=restrict_lower)

    selected_data_upper = load(cosmological_volume, mask=mask_region_upper)
    selected_data_lower = load(cosmological_volume, mask=mask_region_lower)

    selected_coordinates_upper = selected_data_upper.gas.coordinates
    selected_coordinates_lower = selected_data_lower.gas.coordinates
    assert selected_coordinates_lower.size > 0  # check that we selected something
    assert np.array_equal(selected_coordinates_upper, selected_coordinates_lower)


def test_mask_padding(cosmological_volume):
    """
    Check that the padding of a mask when we don't have cell bounding box metadata
    works correctly.
    """

    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    # Mask off the lower bottom corner of the volume.
    if not has_cell_bbox:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_pad_onecell = mask(
                cosmological_volume, spatial_only=True, safe_padding=1.0
            )
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_pad_fifthcell = mask(
                cosmological_volume, spatial_only=True
            )  # default 0.2
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_pad_off = mask(
                cosmological_volume, spatial_only=True, safe_padding=False
            )
    else:
        mask_pad_onecell = mask(
            cosmological_volume, spatial_only=True, safe_padding=1.0
        )
        mask_pad_fifthcell = mask(cosmological_volume, spatial_only=True)  # default 0.2
        mask_pad_off = mask(cosmological_volume, spatial_only=True, safe_padding=False)
    assert mask_pad_onecell.safe_padding == 1.0
    assert mask_pad_fifthcell.safe_padding == 0.2
    assert mask_pad_off.safe_padding == 0.0

    cell_size = mask_pad_onecell.cell_size
    region = cosmo_array([np.ones(3) * 3.8 * cell_size, np.ones(3) * 4.0 * cell_size]).T
    mask_pad_onecell.constrain_spatial(region)
    mask_pad_fifthcell.constrain_spatial(region)
    mask_pad_off.constrain_spatial(region)

    if has_cell_bbox:
        # We should ignore `safe_padding` and just read the cell.
        # Note in case this test fails on a new snapshot:
        # the assertions assume that the neighbouring cells don't
        # have particles that have drifted into the target cell.
        # For a different test snapshot this might not be true,
        # so check for that if troubleshooting.
        assert mask_pad_onecell.cell_mask["gas"].sum() == 1
        assert mask_pad_fifthcell.cell_mask["gas"].sum() == 1
        assert mask_pad_off.cell_mask["gas"].sum() == 1
    else:
        # Padding by a cell length, we should read all 3x3x3 neighbours.
        assert mask_pad_onecell.cell_mask["gas"].sum() == 27
        # Padding by a half-cell length, we should read 2x2x2 cells near this corner.
        assert mask_pad_fifthcell.cell_mask["gas"].sum() == 8
        # Padding switched off, read only this cell.
        assert mask_pad_off.cell_mask["gas"].sum() == 1


def test_mask_pad_wrapping(cosmological_volume):
    """
    When we mask all the way to the edge of the box, we should get a cell on the
    opposite edge as padding in case particles have drifted out of their cell,
    unless the cell metadata with max positions is present.
    """

    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        mask_region_upper = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region_upper = mask(cosmological_volume, spatial_only=True)
    if has_cell_bbox:
        mask_region_lower = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region_lower = mask(cosmological_volume, spatial_only=True)
    restrict_lower = cosmo_array(
        [mask_region_lower.metadata.boxsize * 0.8, mask_region_lower.metadata.boxsize]
    ).T
    restrict_upper = cosmo_array(
        [
            mask_region_upper.metadata.boxsize * 0,
            mask_region_upper.metadata.boxsize * 0.2,
        ]
    ).T

    mask_region_lower.constrain_spatial(restrict=restrict_lower)
    mask_region_upper.constrain_spatial(restrict=restrict_upper)
    selected_data_lower = load(cosmological_volume, mask=mask_region_lower)
    selected_data_upper = load(cosmological_volume, mask=mask_region_upper)
    selected_coordinates_lower = selected_data_lower.gas.coordinates
    selected_coordinates_upper = selected_data_upper.gas.coordinates
    with h5py.File(cosmological_volume, "r") as f:
        if (
            "MinPositions" in f["/Cells"].keys()
            and "MaxPositions" in f["/Cells"].keys()
        ):
            # in the sample files with this metadata present, we should only pick up
            # a small number of extra cells with particles that drifted into our
            # region.
            def expected(n):
                return n < 2000

        else:
            # extending upper mask to a padding cell on the other side of the box
            # should get a lot of particles
            def expected(n):
                return n > 10000

    assert expected(
        (selected_coordinates_lower < mask_region_lower.metadata.boxsize * 0.1).sum()
    )
    assert expected(
        (selected_coordinates_upper > mask_region_upper.metadata.boxsize * 0.9).sum()
    )


def test_mask_entire_box(cosmological_volume):
    """
    When we explicitly set the region to the whole box, we'd better get all of the cells!
    """
    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        mask_region = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region = mask(cosmological_volume, spatial_only=True)
    restrict = cosmo_array(
        [mask_region.metadata.boxsize * 0.0, mask_region.metadata.boxsize]
    ).T

    mask_region.constrain_spatial(restrict=restrict)
    for group_mask in mask_region.cell_mask.values():
        assert group_mask.all()


def test_invalid_mask_interval(cosmological_volume):
    """
    We should get an error if the mask boundaries go out of bounds.
    """
    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        mask_region = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region = mask(cosmological_volume, spatial_only=True)
    restrict = cosmo_array(
        [mask_region.metadata.boxsize * -2, mask_region.metadata.boxsize * 2]
    ).T
    with pytest.raises(ValueError, match="Mask region boundaries must be in interval"):
        mask_region.constrain_spatial(restrict=restrict)


def test_inverted_mask_boundaries(cosmological_volume):
    """
    Upper boundary can be below lower boundary, in that case we select wrapping
    in the other direction. Check this by making an "inverted" selection and
    comparing to the "uninverted" selection through the boundary.
    """
    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        mask_region = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region = mask(cosmological_volume, spatial_only=True)
    if has_cell_bbox:
        mask_region_inverted = mask(cosmological_volume, spatial_only=True)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            mask_region_inverted = mask(cosmological_volume, spatial_only=True)
    restrict = cosmo_array(
        [-mask_region.metadata.boxsize * 0.2, mask_region.metadata.boxsize * 0.2]
    ).T
    restrict_inverted = cosmo_array(
        [mask_region.metadata.boxsize * 0.8, mask_region.metadata.boxsize * 0.2]
    ).T

    mask_region.constrain_spatial(restrict=restrict)
    mask_region_inverted.constrain_spatial(restrict=restrict_inverted)
    selected_data = load(cosmological_volume, mask=mask_region)
    selected_data_inverted = load(cosmological_volume, mask=mask_region_inverted)

    selected_coordinates = selected_data.gas.coordinates
    selected_coordinates_inverted = selected_data_inverted.gas.coordinates
    assert selected_coordinates.size > 0  # check that we selected something
    assert np.array_equal(selected_coordinates, selected_coordinates_inverted)


def test_empty_mask(cosmological_volume):  # replace with cosmoogical_volume_no_legacy
    """
    Tests that a mask containing no particles doesn't cause any problems.
    """
    with h5py.File(cosmological_volume, "r") as f:
        has_cell_bbox = "MinPositions" in f["/Cells"].keys()
    if has_cell_bbox:
        empty_mask = mask(cosmological_volume, spatial_only=False)
    else:
        with pytest.warns(
            UserWarning, match="Snapshot does not contain Cells/MinPositions"
        ):
            empty_mask = mask(cosmological_volume, spatial_only=False)
    # mask a region just to run faster:
    region = [[0 * b, 0.1 * b] for b in empty_mask.metadata.boxsize]
    empty_mask.constrain_spatial(region)
    # pick some values that we'll never find:
    empty_mask.constrain_mask(
        "gas",
        "pressures",
        cosmo_quantity(
            1e59,
            u.solMass * u.Gyr ** -2 * u.Mpc ** -1,
            comoving=False,
            scale_factor=empty_mask.metadata.a,
            scale_exponent=-5,
        ),
        cosmo_quantity(
            1e60,
            u.solMass * u.Gyr ** -2 * u.Mpc ** -1,
            comoving=False,
            scale_factor=empty_mask.metadata.a,
            scale_exponent=-5,
        ),
    )
    data = load(cosmological_volume, mask=empty_mask)
    assert data.gas.masses.size == 0
