"""
Tests the masking using some test data.
"""

from tests.helper import requires
from swiftsimio import load, mask

from unyt import unyt_array as array


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

    restrict = array(
        [[0.0, 0.0, 0.0] * full_data.metadata.boxsize, full_data.metadata.boxsize * 0.5]
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

    full_data = load(filename)

    # Mask off the lower bottom corner of the volume.
    mask_region = mask(filename, spatial_only=True)

    restrict = array(
        [
            [0.0, 0.0, 0.0] * full_data.metadata.boxsize,
            full_data.metadata.boxsize * 0.49,
        ]
    ).T

    mask_region.constrain_spatial(restrict=restrict)

    selected_data = load(filename, mask=mask_region)

    selected_coordinates = selected_data.gas.coordinates

    # Some of these particles will be outside because of the periodic BCs
    assert ((selected_coordinates / full_data.metadata.boxsize) > 0.5).sum() < 25


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

