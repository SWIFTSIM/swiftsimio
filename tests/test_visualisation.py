from swiftsimio.visualisation import scatter, slice, volume_render


try:
    from matplotlib.pyplot import imsave
except:
    pass


def test_scatter(save=False):
    image = scatter(
        [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.2, 0.2, 0.2], 256
    )

    if save:
        imsave("test_image_creation.png", image)

    return


def test_slice(save=False):
    image = slice(
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.2, 0.2, 0.2],
        0.99,
        256,
    )

    if save:
        imsave("test_image_creation.png", image)

    return


def test_volume_render():
    image = volume_render.scatter(
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.2, 0.2, 0.2],
        256,
    )

    return
