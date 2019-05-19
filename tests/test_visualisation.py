from swiftsimio.visualisation import scatter
from matplotlib.pyplot import imsave


def test_scatter(save=False):
    image = scatter(
        [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.2, 0.2, 0.2], 256
    )

    if save:
        imsave("test_image_creation.png", image)

    return
