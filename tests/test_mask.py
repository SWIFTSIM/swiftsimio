"""
Tests the masking using some test data.
"""

from tests.helper import requires
from swiftsimio import load, mask
import numpy as np

from unyt import dimensionless
from swiftsimio import cosmo_array


@requires("cosmological_volume.hdf5")
def test_reading_select_region_spatial(filename):
    """
    Tests reading select regions of the volume, comparing the masks attained with
    spatial_only = True and spatial_only = False.
    """

    full_data = load(filename)

    # Mask off the lower bottom corner of the volume.
    mask_region = mask(filename, spatial_only=True)
    mask_region_nospatial = mask(filename, spatial_only=False)

    restrict = cosmo_array(
        [np.zeros_like(full_data.metadata.boxsize), full_data.metadata.boxsize * 0.5]
    ).T

    mask_region.constrain_spatial(restrict=restrict)
    mask_region_nospatial.constrain_spatial(restrict=restrict)

    selected_data = load(filename, mask=mask_region)
    selected_data_nospatial = load(filename, mask=mask_region_nospatial)

    selected_coordinates = selected_data.gas.coordinates
    selected_coordinates_nospatial = selected_data_nospatial.gas.coordinates

    assert (selected_coordinates_nospatial == selected_coordinates).all()

    return


@requires("cosmological_volume.hdf5")
def test_reading_select_region_half_box(filename):
    """
    Tests reading the spatial region and sees if it lies within the region
    we told it to!

    Specifically, we test to see if all particles lie within half a boxsize.
    """

    # Mask off the lower bottom corner of the volume.
    mask_region = mask(filename, spatial_only=True)

    # the region can be padded by a cell if min & max particle positions are absent
    # in metadata
    pad_frac = (mask_region.cell_size / mask_region.metadata.boxsize).to_value(
        dimensionless
    )
    restrict = cosmo_array(
        [
            mask_region.metadata.boxsize * (pad_frac + 0.01),
            mask_region.metadata.boxsize * (0.5 - pad_frac - 0.01),
        ]
    ).T

    mask_region.constrain_spatial(restrict=restrict)

    selected_data = load(filename, mask=mask_region)

    selected_coordinates = selected_data.gas.coordinates
    # Some of these particles will be outside because of the periodic BCs
    assert (
        (selected_coordinates / mask_region.metadata.boxsize).to_value(dimensionless)
        > 0.5 + pad_frac
    ).sum() < 25
    # make sure the test isn't trivially passed because we selected nothing:
    assert selected_coordinates.size > 0


@requires("cosmological_volume.hdf5")
def test_region_mask_not_modified(filename):
    """
    Tests if a mask region is modified during the course of its use.

    Checks if https://github.com/SWIFTSIM/swiftsimio/issues/22 is broken.
    """

    this_mask = mask(filename, spatial_only=True)
    bs = this_mask.metadata.boxsize

    read = [[0 * b, 0.5 * b] for b in bs]
    read_constant = [[0 * b, 0.5 * b] for b in bs]

    this_mask._generate_cell_mask(read)

    assert read == read_constant


@requires("cosmological_volume.hdf5")
def test_region_mask_intersection(filename):
    """
    Tests that the intersection of two spatial mask regions includes the same cells as two
    separate masks of the same two regions.
    """

    mask_1 = mask(filename, spatial_only=True)
    mask_2 = mask(filename, spatial_only=True)
    mask_intersect = mask(filename, spatial_only=True)
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


@requires("cosmological_volume.hdf5")
def test_mask_periodic_wrapping(filename):
    """
    Check that a region that runs off the upper edge of the box gives the same
    mask as one that runs off the lower edge (they are chosen to be equivalent
    under periodic wrapping).
    """
    mask_region_upper = mask(filename, spatial_only=True)
    mask_region_lower = mask(filename, spatial_only=True)
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

    selected_data_upper = load(filename, mask=mask_region_upper)
    selected_data_lower = load(filename, mask=mask_region_lower)

    selected_coordinates_upper = selected_data_upper.gas.coordinates
    selected_coordinates_lower = selected_data_lower.gas.coordinates
    assert selected_coordinates_lower.size > 0  # check that we selected something
    assert np.array_equal(selected_coordinates_upper, selected_coordinates_lower)


@requires("cosmological_volume.hdf5")
def test_mask_pad_wrapping(filename):
    """
    When we mask all the way to the edge of the box, we should get a cell on the
    opposite edge as padding in case particles have drifted out of their cell.
    """
    mask_region = mask(filename, spatial_only=True)
    restrict = cosmo_array(
        [mask_region.metadata.boxsize * 0.8, mask_region.metadata.boxsize]
    ).T

    mask_region.constrain_spatial(restrict=restrict)
    selected_data = load(filename, mask=mask_region)
    selected_coordinates = selected_data.gas.coordinates
    # extending mask to to 1.01 times the box size gives >32k particles
    # we expect to get the same
    assert (selected_coordinates < mask_region.metadata.boxsize * 0.1).sum() > 32000
