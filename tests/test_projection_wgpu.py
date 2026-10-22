"""Numerical and optional-dependency checks for the WebGPU projection backend."""

import subprocess
import sys

import numpy as np
import pytest

from swiftsimio.visualisation.projection_backends import backends, backends_parallel
from swiftsimio.visualisation.projection_backends.kernels import kernel_gamma
from swiftsimio.visualisation.projection_backends.wgpu import scatter


def test_lazy_device():
    """Check that importing projection backends does not request a GPU device."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
try:
    import wgpu
except ImportError:
    pass
else:
    def fail(*args, **kwargs):
        raise AssertionError('Import requested a GPU adapter')
    wgpu.gpu.request_adapter_sync = fail
import swiftsimio.visualisation.projection_backends
""",
        ],
        check=True,
    )


def test_missing_optional_package():
    """Check that a missing wgpu package is handled by optional package imports."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["wgpu"] = None
from swiftsimio.optional_packages import WGPU_AVAILABLE, wgpu
assert not WGPU_AVAILABLE
assert wgpu is None
from swiftsimio.visualisation.projection_backends.wgpu import scatter
try:
    scatter([0.5], [0.5], [1.0], [0.1], 16)
except ImportError as error:
    assert 'swiftsimio[wgpu]' in str(error)
else:
    raise AssertionError('Missing dependency did not raise ImportError')
""",
        ],
        check=True,
    )


def test_registration_and_empty():
    """Check backend registration and the shape and dtype of an empty result."""
    assert backends["wgpu"] is backends_parallel["wgpu"] is scatter
    result = scatter([], [], [], [], 17)
    assert result.shape == (17, 17)
    assert result.dtype == np.float32
    assert not result.any()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(res=0),
        dict(res=1.5),
        dict(h=[-1]),
        dict(x=[np.nan]),
        dict(m=[1, 2]),
        dict(y=[[0.5]]),
        dict(box_x=-1),
    ],
)
def test_invalid_input(kwargs):
    """
    Check that invalid inputs are rejected before creating GPU resources.

    Parameters
    ----------
    kwargs : dict
        Invalid arguments to override in an otherwise valid projection call.
    """
    args = dict(x=[0.5], y=[0.5], m=[1], h=[0.1], res=32)
    args.update(kwargs)
    with pytest.raises(ValueError):
        scatter(**args)


@pytest.mark.usefixtures("wgpu_device")
@pytest.mark.parametrize("box", [(0.0, 0.0), (1.0, 1.0), (0.4, 0.7), (0.0, 0.6)])
def test_matches_kernel(box):
    """
    Compare splats to analytic kernels, including tiling and axis orientation.

    Parameters
    ----------
    box : tuple[float, float]
        Periodic box sizes along the x and y axes. Zero disables wrapping.
    """
    res = 65  # Also exercise readback with unaligned row lengths.
    x = np.array([0.03, 0.72, 1.04])
    y = np.array([0.63, 0.22, 0.81])
    m = np.array([2.0, -0.7, 1.1])
    h = np.array([0.08, 0.045, 0.09])
    actual = scatter(x, y, m, h, res, *box)
    assert actual.shape == (res, res)
    assert actual.dtype == np.float32
    xx, yy = np.meshgrid(
        (np.arange(res) + 0.5) / res, (np.arange(res) + 0.5) / res, indexing="ij"
    )
    expected = np.zeros((res, res))
    shifts = [range(1) if b == 0 else range(-1, int(np.ceil(1 / b)) + 1) for b in box]
    for px, py, weight, hsml in zip(x, y, m, h):
        radius = float(kernel_gamma) * hsml
        for dx in shifts[0]:
            for dy in shifts[1]:
                q = np.hypot(xx - px - dx * box[0], yy - py - dy * box[1]) / radius
                t = np.maximum(1 - q, 0)
                expected += weight * 2.22817109 / radius**2 * t**4 * (1 + 4 * q)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-4)


@pytest.mark.usefixtures("wgpu_device")
def test_unresolved_and_repeat():
    """Check unresolved kernel weights and clear the output between calls."""
    for res in (32, 65):
        image = scatter([0.3], [0.7], [3.0], [0.0], res)
        assert np.isfinite(image).all()
        assert image.sum() / res**2 == pytest.approx(3.0, rel=0.05)
        np.testing.assert_array_equal(
            scatter([0.3], [0.7], [-3.0], [1e-12], res), -image
        )


@pytest.mark.usefixtures("wgpu_device")
@pytest.mark.parametrize("parallel", [False, True])
def test_project_gas(tmp_path, parallel):
    """
    Check units, cosmology, masks and rectangular cropping through the API.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for the test snapshot.
    parallel : bool
        Whether to select the parallel projection backend.
    """
    from swiftsimio import load
    from swiftsimio.visualisation.projection import project_gas
    from .helper import create_minimal_writer

    writer = create_minimal_writer(a=0.5, n_p=8, lbox=1.0)
    writer.gas.coordinates[:] = np.random.default_rng(42).uniform(0.2, 0.8, (8, 3))
    writer.gas.smoothing_lengths[:] = 0.08
    filename = str(tmp_path / "projection.hdf5")
    writer.write(filename)
    data = load(filename)
    kwargs = dict(
        resolution=65,
        region=data.metadata.boxsize[0] * np.array([0, 1, 0, 0.6]),
        mask=np.arange(8) % 2 == 0,
        periodic=False,
    )
    actual = project_gas(data, backend="wgpu", parallel=parallel, **kwargs)
    expected = project_gas(data, backend="fast", **kwargs)
    assert actual.shape == (65, 39)
    assert actual.units == expected.units
    assert actual.comoving == expected.comoving
    assert actual.cosmo_factor == expected.cosmo_factor
    np.testing.assert_allclose(
        actual.value, expected.value, rtol=3e-4, atol=expected.max().value * 1e-6
    )


@pytest.mark.parametrize("scientific_first", [False, True])
def test_scientific_import_compatibility(scientific_first):
    """
    Check that wgpu and scientific dependencies can be imported in either order.

    Parameters
    ----------
    scientific_first : bool
        Whether to import Astropy and h5py before wgpu.
    """
    pytest.importorskip("wgpu")
    imports = ["import wgpu", "from astropy.table import Table; import h5py"]
    if scientific_first:
        imports.reverse()
    code = "import sys\nsys.modules['pygfx'] = None\nsys.modules['uharfbuzz'] = None\n"
    code += "\n".join(imports)
    code += "\nfrom swiftsimio.optional_packages import WGPU_AVAILABLE\n"
    code += "assert WGPU_AVAILABLE\n"
    subprocess.run([sys.executable, "-c", code], check=True)
