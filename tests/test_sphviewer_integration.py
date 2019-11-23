"""
Uses test data to test the py-sphviewer integration.
"""

from tests.helper import requires
from swiftsimio import load, mask
from swiftsimio.visualisation.sphviewer import SPHViewerWrapper

from unyt import unyt_array as array

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except (ImportError, ModuleNotFoundError):
    pass

create_plots = False


@requires("cosmological_volume.hdf5")
def test_gas_rendering(filename, create_plots=create_plots):
    data = load(filename)

    data.gas.coordinates.convert_to_units("Mpc")
    data.gas.smoothing_lengths.convert_to_units("Mpc")

    wrap = SPHViewerWrapper(data.gas)

    wrap.quick_view(xsize=512, ysize=512, r="infinity")

    if create_plots:
        plt.imshow(wrap.image.value, extent=wrap.extent.value, origin="lower")

        plt.savefig("test_pysphviewer_integration.png")


@requires("cosmological_volume.hdf5")
def test_dm_rendering(filename, create_plots=create_plots):
    data = load(filename)

    data.dark_matter.coordinates.convert_to_units("Mpc")

    wrap = SPHViewerWrapper(data.dark_matter)

    wrap.quick_view(xsize=512, ysize=512, r="infinity")

    if create_plots:
        plt.imshow(wrap.image.value, extent=wrap.extent.value, origin="lower")

        plt.savefig("test_pysphviewer_integration.png")
