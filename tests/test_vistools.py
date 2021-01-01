"""
Tests the visualisation tools.
"""

from swiftsimio.visualisation.tools.cmaps import LinearSegmentedCmap2D
import numpy as np


def test_basic_linear_2d():
    x = LinearSegmentedCmap2D(
        colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        coordinates=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    x.color_map_grid

    return


def test_apply_to_data_2d():
    x = LinearSegmentedCmap2D(
        colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        coordinates=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    x(np.linspace(0, 1, 10), np.linspace(0.2, 0.8, 10))

    return
