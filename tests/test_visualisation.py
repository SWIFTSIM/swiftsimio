from swiftsimio.visualisation import scatter, slice, volume_render
from swiftsimio.visualisation.projection import scatter_parallel
from swiftsimio.visualisation.slice import slice_scatter_parallel

import numpy as np


try:
    from matplotlib.pyplot import imsave
except:
    pass


def test_scatter(save=False):
    image = scatter(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.2, 0.2, 0.2]),
        256,
    )

    if save:
        imsave("test_image_creation.png", image)

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
        256,
    )

    return


def test_volume_parallel():
    number_of_parts = 1000
    h_max = np.float32(0.05)
    resolution = 128

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
