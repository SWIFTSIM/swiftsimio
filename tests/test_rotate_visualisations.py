from tests.helper import requires, create_single_particle_dataset
from swiftsimio import load
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.volume_render import render_gas
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from numpy import array_equal
from os import remove


@requires("cosmological_volume.hdf5")
def test_project(filename):
    """
    Checks that gas projection of a single particle snapshot is invariant under
    rotations around the particle

    Parameters
    ----------
    filename: str
        name of file providing metadata to copy
    """
    # Start from the beginning, open the file
    output_filename = "single_particle.hdf5"
    create_single_particle_dataset(filename, output_filename)
    data = load(output_filename)

    unrotated = project_gas(
        data, resolution=1024, project="masses", parallel=True, periodic=False
    )

    # Compute rotation matrix for rotating around particle
    centre = data.gas.coordinates[0]
    rotate_vec = [0.5, 0.5, 0.5]
    matrix = rotation_matrix_from_vector(rotate_vec, axis="z")
    rotated = project_gas(
        data,
        resolution=1024,
        project="masses",
        rotation_center=centre,
        rotation_matrix=matrix,
        parallel=True,
        periodic=False,
    )

    assert array_equal(rotated, unrotated)

    remove(output_filename)


@requires("cosmological_volume.hdf5")
def test_slice(filename):
    """
    Checks that a slice of a single particle snapshot is invariant under
    rotations around the particle

    Parameters
    ----------
    filename: str
        name of file providing metadata to copy
    """
    # Start from the beginning, open the file
    output_filename = "single_particle.hdf5"
    create_single_particle_dataset(filename, output_filename)
    data = load(output_filename)

    unrotated = slice_gas(
        data,
        resolution=1024,
        z_slice=data.gas.coordinates[0, 2],
        project="masses",
        parallel=True,
        periodic=False,
    )

    # Compute rotation matrix for rotating around particle
    centre = data.gas.coordinates[0]
    rotate_vec = [0.5, 0.5, 0.5]
    matrix = rotation_matrix_from_vector(rotate_vec, axis="z")
    rotated = slice_gas(
        data,
        resolution=1024,
        z_slice=0 * data.gas.coordinates[0, 2],
        project="masses",
        rotation_center=centre,
        rotation_matrix=matrix,
        parallel=True,
        periodic=False,
    )

    # Check that we didn't miss the particle
    assert unrotated.any()
    assert rotated.any()

    assert array_equal(rotated, unrotated)

    remove(output_filename)


@requires("cosmological_volume.hdf5")
def test_render(filename):
    """
    Checks that a volume render of a single particle snapshot is invariant under
    rotations around the particle

    Parameters
    ----------
    filename: str
        name of file providing metadata to copy
    """
    # Start from the beginning, open the file
    output_filename = "single_particle.hdf5"
    create_single_particle_dataset(filename, output_filename)
    data = load(output_filename)

    unrotated = render_gas(
        data, resolution=256, project="masses", parallel=True, periodic=False
    )

    # Compute rotation matrix for rotating around particle
    centre = data.gas.coordinates[0]
    rotate_vec = [0.5, 0.5, 0.5]
    matrix = rotation_matrix_from_vector(rotate_vec, axis="z")
    rotated = render_gas(
        data,
        resolution=256,
        project="masses",
        rotation_center=centre,
        rotation_matrix=matrix,
        parallel=True,
        periodic=False,
    )

    assert array_equal(rotated, unrotated)

    remove(output_filename)
