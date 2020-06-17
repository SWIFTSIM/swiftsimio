from tests.helper import requires, create_single_particle_dataset
from swiftsimio import load
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.volume_render import render_gas
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from numpy import array_equal
from os import remove
import pytest


@pytest.mark.skip()
def test_project(data, centre, matrix):
    """
    Checks that gas projection of a single particle dataset is invariant under
    rotations around the particle

    Parameters
    ----------
    data: SWIFTDataset
        SWIFTDataset containing a single particle
    centre: cosmo_array
        Point around which to rotate
    matrix: np.ndarray
        rotation matrix specifying the rotation transformation
    """
    unrotated = project_gas(data, resolution=1024, project="masses", parallel=True)

    rotated = project_gas(
        data,
        resolution=1024,
        project="masses",
        rotation_center=centre,
        rotation_matrix=matrix,
        parallel=True,
    )

    assert array_equal(rotated, unrotated)

    return


@pytest.mark.skip()
def test_slice(data, slice_z, centre, matrix):
    """
    Checks that a slice of a single particle dataset is invariant under
    rotations around the particle

    Parameters
    ----------
    data: SWIFTDataset
        SWIFTDataset containing a single particle
    slice_z: float
        location along z-axis where the slice is taken
    centre: cosmo_array
        Point around which to rotate
    matrix: np.ndarray
        rotation matrix specifying the rotation transformation
    """
    unrotated = slice_gas(
        data, resolution=1024, slice=slice_z, project="masses", parallel=True
    )

    rotated = slice_gas(
        data,
        resolution=1024,
        slice=slice_z,
        project="masses",
        rotation_center=centre,
        rotation_matrix=matrix,
        parallel=True,
    )

    # Check that we didn't miss the particle
    assert unrotated.any()
    assert rotated.any()

    assert array_equal(rotated, unrotated)


@pytest.mark.skip()
def test_render(data, centre, matrix):
    """
    Checks that a volume render of a single particle dataset is invariant under
    rotations around the particle

    Parameters
    ----------
    data: SWIFTDataset
        SWIFTDataset containing a single particle
    centre: cosmo_array
        Point around which to rotate
    matrix: np.ndarray
        rotation matrix specifying the rotation transformation
    """
    unrotated = render_gas(data, resolution=256, project="masses", parallel=True)

    rotated = render_gas(
        data,
        resolution=256,
        project="masses",
        rotation_center=centre,
        rotation_matrix=matrix,
        parallel=True,
    )

    assert array_equal(rotated, unrotated)


@requires("cosmological_volume.hdf5")
def test_rotation(filename):
    """
    Check that projection, slicing and volume rendering of a single particle
    are invariant under rotations around the particle's location.

    To do this a snapshot with a single particle is created, reusing most of the
    metadata from the cosmological volume snapshot.

    Parameters
    ----------
    filename: str
        name of file providing metadata to copy
    """

    # Start from the beginning, open the file
    output_filename = "single_particle.hdf5"
    create_single_particle_dataset(filename, output_filename)
    data = load(output_filename)

    # Compute rotation matrix for rotating around particle
    centre = data.gas.coordinates[0]
    rotate_vec = [0.5, 0.5, 0.5]
    matrix = rotation_matrix_from_vector(rotate_vec, axis="z")
    boxsize = data.metadata.boxsize

    # Check the projection first
    test_project(data, centre, matrix)

    # Now check rotations in slices
    # First find the locations of the slices that contain our particle
    z_range = boxsize[2]
    slice_z = centre[2] / z_range

    test_slice(data, slice_z, centre, matrix)

    # And now check the volume render

    remove(output_filename)
