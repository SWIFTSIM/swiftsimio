from swiftsimio import load
from swiftsimio.visualisation import scatter, slice, volume_render
from swiftsimio.visualisation.projection import scatter_parallel, project_gas
from swiftsimio.visualisation.slice import slice_scatter_parallel, slice_gas
from swiftsimio.visualisation.projection_backends import backends, backends_parallel

from tests.helper import requires

import numpy as np


try:
    from matplotlib.pyplot import imsave
except:
    pass


def test_scatter(save=False):
    """
    Tests the scatter functions from all backends.
    """

    for backend in backends.values():
        image = backend(
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.2, 0.2, 0.2]),
            256,
        )

    if save:
        imsave("test_image_creation.png", image)

    return


def test_scatter_mass_conservation():
    np.random.seed(971263)
    # Width of 0.8 centered on 0.5, 0.5.
    x = 0.8 * np.random.rand(100) + 0.1
    y = 0.8 * np.random.rand(100) + 0.1
    m = np.ones_like(x)
    h = 0.05 * np.ones_like(x)

    resolutions = [8, 16, 32, 64, 128, 256, 512]
    total_mass = np.sum(m)

    for resolution in resolutions:
        image = scatter(x, y, m, h, resolution)
        mass_in_image = image.sum() / (resolution ** 2)

        # Check mass conservation to 5%
        assert np.isclose(mass_in_image, total_mass, 0.05)

    return


def test_scatter_parallel(save=False):
    """
    Asserts that we create the same image with the parallel version of the code
    as with the serial version.
    """

    number_of_parts = 1000
    h_max = np.float32(0.05)
    resolution = 512

    coordinates = (
        np.random.rand(2 * number_of_parts)
        .reshape((2, number_of_parts))
        .astype(np.float64)
    )
    hsml = np.random.rand(number_of_parts).astype(np.float32) * h_max
    masses = np.ones(number_of_parts, dtype=np.float32)

    image = scatter(coordinates[0], coordinates[1], masses, hsml, resolution)
    image_par = scatter_parallel(
        coordinates[0], coordinates[1], masses, hsml, resolution
    )

    if save:
        imsave("test_image_creation.png", image)

    assert np.isclose(image, image_par).all()

    return


def test_slice(save=False):
    image = slice(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.2, 0.2, 0.2]),
        0.99,
        256,
    )

    if save:
        imsave("test_image_creation.png", image)

    return


def test_slice_parallel(save=False):
    """
    Asserts that we create the same image with the parallel version of the code
    as with the serial version.
    """

    number_of_parts = 1000
    h_max = np.float32(0.05)
    z_slice = 0.5
    resolution = 256

    coordinates = (
        np.random.rand(3 * number_of_parts)
        .reshape((3, number_of_parts))
        .astype(np.float64)
    )
    hsml = np.random.rand(number_of_parts).astype(np.float32) * h_max
    masses = np.ones(number_of_parts, dtype=np.float32)

    image = slice(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        masses,
        hsml,
        z_slice,
        resolution,
    )
    image_par = slice_scatter_parallel(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        masses,
        hsml,
        z_slice,
        resolution,
    )

    if save:
        imsave("test_image_creation.png", image)

    assert np.isclose(image, image_par).all()

    return


def test_volume_render():
    image = volume_render.scatter(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.2, 0.2, 0.2]),
        64,
    )

    return


def test_volume_parallel():
    number_of_parts = 1000
    h_max = np.float32(0.05)
    resolution = 64

    coordinates = (
        np.random.rand(3 * number_of_parts)
        .reshape((3, number_of_parts))
        .astype(np.float64)
    )
    hsml = np.random.rand(number_of_parts).astype(np.float32) * h_max
    masses = np.ones(number_of_parts, dtype=np.float32)

    image = volume_render.scatter(
        coordinates[0], coordinates[1], coordinates[2], masses, hsml, resolution
    )
    image_par = volume_render.scatter_parallel(
        coordinates[0], coordinates[1], coordinates[2], masses, hsml, resolution
    )

    assert np.isclose(image, image_par).all()

    return


@requires("cosmological_volume.hdf5")
def test_selection_render(filename):
    data = load(filename)
    bs = data.metadata.boxsize[0]

    # Projection
    render_full = project_gas(data, 256, parallel=True)
    render_partial = project_gas(
        data, 256, parallel=True, region=[0.25 * bs, 0.75 * bs] * 2
    )
    render_tiny = project_gas(data, 256, parallel=True, region=[0 * bs, 0.001 * bs] * 2)

    # Slicing
    render_full = slice_gas(data, 256, slice=0.5, parallel=True)
    render_partial = slice_gas(
        data, 256, slice=0.5, parallel=True, region=[0.25 * bs, 0.75 * bs] * 2
    )
    render_tiny = slice_gas(
        data, 256, slice=0.5, parallel=True, region=[0 * bs, 0.001 * bs] * 2
    )

    # If they don't crash we're happy!

    return


def test_render_outside_region():
    """
    Tests what happens when you use `scatter` on a bunch of particles that live
    outside of the image.
    """

    number_of_parts = 10000
    resolution = 256

    x = np.random.rand(number_of_parts) - 0.5
    y = np.random.rand(number_of_parts) - 0.5
    z = np.random.rand(number_of_parts) - 0.5
    h = 10 ** np.random.rand(number_of_parts) - 1.0
    h[h > 0.5] = 0.05
    m = np.ones_like(h)
    backends["histogram"](x, y, m, h, resolution)

    for _, backend in backends_parallel.items():
        backend(x, y, m, h, resolution)

    slice_scatter_parallel(x, y, z, m, h, 0.2, resolution)

    volume_render.scatter_parallel(x, y, z, m, h, resolution)

