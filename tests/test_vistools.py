"""
Tests the visualisation tools.
"""

from swiftsimio.visualisation.tools.cmaps import (
    LinearSegmentedCmap2D,
    LinearSegmentedCmap2DHSV,
)
import numpy as np
from swiftsimio.visualisation._vistools import _get_projection_field
from swiftsimio import load
from tests.helper import requires


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
        return y ** 2

    raster_at = np.linspace(0, 1, 1024)

    x, y = np.meshgrid(horizontal_func(raster_at), vertical_func(raster_at))

    bower(x, y)
    bower_hsv(x, y)

    return


@requires("cosmo_volume_example.hdf5")
def test_get_projection_field(filename):
    sd = load(filename)
    expected_dataset = sd.gas.masses
    obtained_dataset = _get_projection_field(sd.gas, "masses")
    assert np.allclose(expected_dataset, obtained_dataset)
    expected_namedcolumn = sd.gas.element_mass_fractions.carbon
    obtained_namedcolumn = _get_projection_field(
        sd.gas, "element_mass_fractions.carbon"
    )
    assert np.allclose(expected_namedcolumn, obtained_namedcolumn)
