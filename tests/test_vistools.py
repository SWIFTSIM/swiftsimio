"""
Tests the visualisation tools.
"""

from swiftsimio.visualisation.tools.cmaps import (
    LinearSegmentedCmap2D,
    LinearSegmentedCmap2DHSV,
)
import numpy as np


def test_basic_linear_2d():
    x = LinearSegmentedCmap2D(
        colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        coordinates=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    x.color_map_grid

    return


def test_apply_to_data_2d():
    bower = LinearSegmentedCmap2D(
        colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        coordinates=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    bower_hsv = LinearSegmentedCmap2DHSV(
        colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        coordinates=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    def vertical_func(x):
        return abs(1.0 - 2.0 * x)

    def horizontal_func(y):
        return y**2

    raster_at = np.linspace(0, 1, 1024)

    x, y = np.meshgrid(horizontal_func(raster_at), vertical_func(raster_at))

    imaged = bower(x, y)
    imaged_hsv = bower_hsv(x, y)

    return
