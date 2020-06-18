from tests.helper import requires, create_single_particle_dataset
from swiftsimio import load
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.volume_render import render_gas
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from numpy import array_equal
from os import remove
import pytest


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

    # Compute rotation matrix for rotating around particle
    centre = data.gas.coordinates[0]
    rotate_vec = [0.5, 0.5, 0.5]
    matrix = rotation_matrix_from_vector(rotate_vec, axis="z")
    boxsize = data.metadata.boxsize

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

    # Compute rotation matrix for rotating around particle
    centre = data.gas.coordinates[0]
    rotate_vec = [0.5, 0.5, 0.5]
    matrix = rotation_matrix_from_vector(rotate_vec, axis="z")
    boxsize = data.metadata.boxsize

    z_range = boxsize[2]
    slice_z = centre[2] / z_range

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

    # Compute rotation matrix for rotating around particle
    centre = data.gas.coordinates[0]
    rotate_vec = [0.5, 0.5, 0.5]
    matrix = rotation_matrix_from_vector(rotate_vec, axis="z")
    boxsize = data.metadata.boxsize

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

    remove(output_filename)
