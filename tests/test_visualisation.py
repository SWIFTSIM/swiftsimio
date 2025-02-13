import pytest
from swiftsimio import load, mask
from swiftsimio.visualisation.projection import project_gas, project_pixel_grid
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.volume_render import render_gas
from swiftsimio.visualisation.ray_trace import panel_gas

from swiftsimio.visualisation.slice_backends import (
    backends as slice_backends,
    backends_parallel as slice_backends_parallel,
)
from swiftsimio.visualisation.volume_render_backends import (
    backends as volume_render_backends,
    backends_parallel as volume_render_backends_parallel,
)
from swiftsimio.visualisation.projection_backends import (
    backends as projection_backends,
    backends_parallel as projection_backends_parallel,
)

from swiftsimio.visualisation.power_spectrum import (
    deposit,
    deposition_to_power_spectrum,
    render_to_deposit,
    folded_depositions_to_power_spectrum,
)
from swiftsimio.visualisation.smoothing_length import generate_smoothing_lengths
from swiftsimio.optional_packages import CudaSupportError, CUDA_AVAILABLE

from swiftsimio.objects import cosmo_array, cosmo_quantity, cosmo_factor, a
from unyt.array import unyt_array
import unyt

from tests.helper import requires

import numpy as np


try:
    from matplotlib.pyplot import imsave
except ImportError:
    pass


def test_scatter(save=False):
    """
    Tests the scatter functions from all backends.
    """

    for backend in projection_backends.keys():
        try:
            image = projection_backends[backend](
                x=np.array([0.0, 1.0, 1.0, -0.000_001]),
                y=np.array([0.0, 0.0, 1.0, 1.000_001]),
                m=np.array([1.0, 1.0, 1.0, 1.0]),
                h=np.array([0.2, 0.2, 0.2, 0.000_002]),
                res=256,
                box_x=1.0,
                box_y=1.0,
            )
        except CudaSupportError:
            if CUDA_AVAILABLE:
                raise ImportError("Optional loading of the CUDA module is broken")
            else:
                continue

    if save:
        imsave("test_image_creation.png", image)

    return


def test_scatter_mass_conservation():
    np.random.seed(971_263)
    # Width of 0.8 centered on 0.5, 0.5.
    x = 0.8 * np.random.rand(100) + 0.1
    y = 0.8 * np.random.rand(100) + 0.1
    m = np.ones_like(x)
    h = 0.05 * np.ones_like(x)

    resolutions = [8, 16, 32, 64, 128, 256, 512]
    total_mass = np.sum(m)

    for resolution in resolutions:
        scatter = projection_backends["fast"]
        image = scatter(x=x, y=y, m=m, h=h, res=resolution, box_x=1.0, box_y=1.0)
        mass_in_image = image.sum() / (resolution ** 2)

        # Check mass conservation to 5%
        assert np.isclose(mass_in_image.view(np.ndarray), total_mass, 0.05)

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

    scatter = projection_backends["fast"]
    scatter_parallel = projection_backends_parallel["fast"]
    image = scatter(
        x=coordinates[0],
        y=coordinates[1],
        m=masses,
        h=hsml,
        res=resolution,
        box_x=1.0,
        box_y=1.0,
    )
    image_par = scatter_parallel(
        x=coordinates[0],
        y=coordinates[1],
        m=masses,
        h=hsml,
        res=resolution,
        box_x=1.0,
        box_y=1.0,
    )

    if save:
        imsave("test_image_creation.png", image)

    assert np.isclose(image, image_par).all()

    return


def test_slice(save=False):
    slice = slice_backends["sph"]
    image = slice(
        x=np.array([0.0, 1.0, 1.0, -0.000_001]),
        y=np.array([0.0, 0.0, 1.0, 1.000_001]),
        z=np.array([0.0, 0.0, 1.0, 1.000_001]),
        m=np.array([1.0, 1.0, 1.0, 1.0]),
        h=np.array([0.2, 0.2, 0.2, 0.000_002]),
        z_slice=0.99,
        xres=256,
        yres=256,
        box_x=1.0,
        box_y=1.0,
        box_z=1.0,
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

    for backend in slice_backends.keys():
        image = slice_backends[backend](
            x=coordinates[0],
            y=coordinates[1],
            z=coordinates[2],
            m=masses,
            h=hsml,
            z_slice=z_slice,
            xres=resolution,
            yres=resolution,
            box_x=1.0,
            box_y=1.0,
            box_z=1.0,
        )
        image_par = slice_backends_parallel[backend](
            x=coordinates[0],
            y=coordinates[1],
            z=coordinates[2],
            m=masses,
            h=hsml,
            z_slice=z_slice,
            xres=resolution,
            yres=resolution,
            box_x=1.0,
            box_y=1.0,
            box_z=1.0,
        )

        assert np.isclose(image, image_par).all()

    if save:
        imsave("test_image_creation.png", image)

    return


def test_volume_render():
    # render image
    scatter = volume_render_backends["scatter"]
    scatter(
        x=np.array([0.0, 1.0, 1.0, -0.000_001]),
        y=np.array([0.0, 0.0, 1.0, 1.000_001]),
        z=np.array([0.0, 0.0, 1.0, 1.000_001]),
        m=np.array([1.0, 1.0, 1.0, 1.0]),
        h=np.array([0.2, 0.2, 0.2, 0.000_002]),
        res=64,
        box_x=1.0,
        box_y=1.0,
        box_z=1.0,
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

    scatter = volume_render_backends["scatter"]
    image = scatter(
        x=coordinates[0],
        y=coordinates[1],
        z=coordinates[2],
        m=masses,
        h=hsml,
        res=resolution,
        box_x=1.0,
        box_y=1.0,
        box_z=1.0,
    )
    scatter_parallel = volume_render_backends_parallel["scatter"]
    image_par = scatter_parallel(
        x=coordinates[0],
        y=coordinates[1],
        z=coordinates[2],
        m=masses,
        h=hsml,
        res=resolution,
        box_x=1.0,
        box_y=1.0,
        box_z=1.0,
    )

    assert np.isclose(image, image_par).all()

    return


@requires("cosmological_volume.hdf5")
def test_selection_render(filename):
    data = load(filename)
    bs = data.metadata.boxsize[0]

    # Projection
    # render full
    project_gas(data, 256, parallel=True)
    # render partial
    project_gas(data, 256, parallel=True, region=[0.25 * bs, 0.75 * bs] * 2)
    # render tiny
    project_gas(data, 256, parallel=True, region=[0 * bs, 0.001 * bs] * 2)
    # render non-square
    project_gas(
        data, 256, parallel=True, region=[0 * bs, 0.50 * bs, 0.25 * bs, 0.75 * bs]
    )

    # Slicing
    # render full
    slice_gas(data, 256, z_slice=0.5 * bs, parallel=True)
    # render partial
    slice_gas(
        data, 256, z_slice=0.5 * bs, parallel=True, region=[0.25 * bs, 0.75 * bs] * 2
    )
    # render tiny
    slice_gas(
        data, 256, z_slice=0.5 * bs, parallel=True, region=[0 * bs, 0.001 * bs] * 2
    )
    # Test for non-square slices
    slice_gas(
        data,
        256,
        z_slice=0.5 * bs,
        parallel=True,
        region=[0 * bs, 0.001 * bs, 0.25 * bs, 0.75 * bs],
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
    projection_backends["histogram"](
        x=x, y=y, m=m, h=h, res=resolution, box_x=1.0, box_y=1.0
    )

    for backend in projection_backends.keys():
        try:
            projection_backends[backend](
                x=x, y=y, m=m, h=h, res=resolution, box_x=1.0, box_y=1.0
            )
        except CudaSupportError:
            if CUDA_AVAILABLE:
                raise ImportError("Optional loading of the CUDA module is broken")
            else:
                continue

    slice_backends_parallel["sph"](
        x=x,
        y=y,
        z=z,
        m=m,
        h=h,
        z_slice=0.2,
        xres=resolution,
        yres=resolution,
        box_x=1.0,
        box_y=1.0,
        box_z=1.0,
    )

    volume_render_backends_parallel["scatter"](
        x=x, y=y, z=z, m=m, h=h, res=resolution, box_x=1.0, box_y=1.0, box_z=1.0
    )


@requires("cosmological_volume.hdf5")
def test_comoving_versus_physical(filename):
    """
    Test what happens if you try to mix up physical and comoving quantities.
    """

    # this test is pretty slow if we don't mask out some particles
    m = mask(filename)
    boxsize = m.metadata.boxsize
    m.constrain_spatial([[0.0 * b, 0.2 * b] for b in boxsize])
    region = [
        0.0 * boxsize[0],
        0.2 * boxsize[0],
        0.0 * boxsize[1],
        0.2 * boxsize[1],
        0.0 * boxsize[2],
        0.2 * boxsize[2],
    ]
    for func, aexp in [(project_gas, -2.0), (slice_gas, -3.0), (render_gas, -3.0)]:
        # normal case: everything comoving
        data = load(filename, mask=m)
        # we force the default (project="masses") to check the cosmo_factor
        # conversion in this case
        img = func(data, resolution=64, project=None, region=region)
        assert data.gas.masses.comoving and img.comoving
        assert (img.cosmo_factor.expr - a ** (aexp)).simplify() == 0
        img = func(data, resolution=64, project="densities", region=region)
        assert data.gas.densities.comoving and img.comoving
        assert (img.cosmo_factor.expr - a ** (aexp - 3.0)).simplify() == 0
        # try to mix comoving coordinates with a physical variable:
        # the coordinates should convert to physical internally and warn
        data.gas.densities.convert_to_physical()
        with pytest.warns(
            UserWarning, match="Converting smoothing lengths to physical."
        ):
            with pytest.warns(
                UserWarning, match="Converting coordinate grid to physical."
            ):
                img = func(data, resolution=64, project="densities", region=region)
        assert data.gas.densities.comoving is False and img.comoving is False
        assert (img.cosmo_factor.expr - a ** (aexp - 3.0)).simplify() == 0
        # convert coordinates to physical (but not smoothing lengths):
        # the coordinates (copy) should convert back to comoving to match the masses
        data.gas.coordinates.convert_to_physical()
        with pytest.warns(UserWarning, match="Converting coordinate grid to comoving."):
            img = func(data, resolution=64, project="masses", region=region)
        assert data.gas.masses.comoving and img.comoving
        assert (img.cosmo_factor.expr - a ** (aexp)).simplify() == 0
        # also convert smoothing lengths to physical
        # everything should still convert back to comoving to match masses
        data.gas.smoothing_lengths.convert_to_physical()
        with pytest.warns(
            UserWarning, match="Converting smoothing lengths to comoving."
        ):
            with pytest.warns(
                UserWarning, match="Converting coordinate grid to comoving."
            ):
                img = func(data, resolution=64, project="masses", region=region)
        assert data.gas.masses.comoving and img.comoving
        assert (img.cosmo_factor.expr - a ** aexp).simplify() == 0
        # densities are physical, make sure this works with physical coordinates and
        # smoothing lengths
        img = func(data, resolution=64, project="densities", region=region)
        assert data.gas.densities.comoving is False and img.comoving is False
        assert (img.cosmo_factor.expr - a ** (aexp - 3.0)).simplify() == 0
        # now try again with comoving densities, should work and give a comoving img
        # with internal conversions to comoving
        data.gas.densities.convert_to_comoving()
        with pytest.warns(
            UserWarning, match="Converting smoothing lengths to comoving."
        ):
            with pytest.warns(
                UserWarning, match="Converting coordinate grid to comoving."
            ):
                img = func(data, resolution=64, project="densities", region=region)
        assert data.gas.densities.comoving and img.comoving
        assert (img.cosmo_factor.expr - a ** (aexp - 3.0)).simplify() == 0


@requires("cosmological_volume.hdf5")
def test_nongas_smoothing_lengths(filename):
    """
    Test that the visualisation tools to calculate smoothing lengths give usable results.
    """

    # If project_gas runs without error the smoothing lengths seem usable.
    data = load(filename)
    data.dark_matter.smoothing_length = generate_smoothing_lengths(
        data.dark_matter.coordinates, data.metadata.boxsize, kernel_gamma=1.8
    )
    project_pixel_grid(data.dark_matter, resolution=256, project="masses")
    assert isinstance(data.dark_matter.smoothing_length, cosmo_array)

    # We should also be able to use a unyt_array (rather than cosmo_array) as input,
    # and in this case get unyt_array as output.
    unyt_array_input = unyt_array(
        data.dark_matter.coordinates.to_value(data.dark_matter.coordinates.units),
        units=data.dark_matter.coordinates.units,
    )
    hsml = generate_smoothing_lengths(
        unyt_array_input, data.metadata.boxsize, kernel_gamma=1.8
    )
    assert isinstance(hsml, unyt_array)
    assert not isinstance(hsml, cosmo_array)

    return


@requires("cosmological_volume.hdf5")
def test_panel_rendering(filename):
    data = load(filename)

    N_depth = 32
    res = 1024

    # Test the panel rendering
    panel = panel_gas(data, resolution=res, panels=N_depth, project="masses")

    assert panel.shape[-1] == N_depth

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.imsave(
        "panels_added.png",
        plt.get_cmap()(LogNorm(vmin=10 ** 6, vmax=10 ** 6.5)(np.sum(panel, axis=-1))),
    )

    projected = project_gas(data, res, "masses", backend="renormalised")

    plt.imsave(
        "projected.png",
        plt.get_cmap()(LogNorm(vmin=10 ** 6, vmax=10 ** 6.5)(projected)),
    )

    fullstack = np.zeros((res, res))

    for i in range(N_depth):
        fullstack = fullstack * 0.5 + panel[:, :, i].v

    offset = 32

    plt.imsave(
        "stacked.png",
        plt.get_cmap()(LogNorm()(fullstack[offset:-offset, offset:-offset])),
    )

    assert np.isclose(
        panel.sum(axis=-1)[offset:-offset, offset:-offset],
        projected[offset:-offset, offset:-offset],
        rtol=0.1,
    ).all()
    return


def test_periodic_boundary_wrapping():
    """
    Test that periodic boundary wrapping works.
    """

    voxel_resolution = 10
    pixel_resolution = 100
    boxsize = 1.0

    # set up a particle near the edge of the box that overlaps with the edge
    coordinates_periodic = np.array([[0.1, 0.5, 0.5]])
    hsml_periodic = np.array([0.2])
    masses_periodic = np.array([1.0])

    # set up a periodic copy of the particle on the other side of the box as well
    # to test the case where we don't apply periodic wrapping
    coordinates_non_periodic = np.array([[0.1, 0.5, 0.5], [1.1, 0.5, 0.5]])
    hsml_non_periodic = np.array([0.2, 0.2])
    masses_non_periodic = np.array([1.0, 1.0])

    # test the projection backends scatter functions
    for backend in projection_backends.keys():
        try:
            image1 = projection_backends[backend](
                x=coordinates_periodic[:, 0],
                y=coordinates_periodic[:, 1],
                m=masses_periodic,
                h=hsml_periodic,
                res=pixel_resolution,
                box_x=boxsize,
                box_y=boxsize,
            )
            image2 = projection_backends[backend](
                x=coordinates_non_periodic[:, 0],
                y=coordinates_non_periodic[:, 1],
                m=masses_non_periodic,
                h=hsml_non_periodic,
                res=pixel_resolution,
                box_x=0.0,
                box_y=0.0,
            )
            assert (image1 == image2).all()
        except CudaSupportError:
            if CUDA_AVAILABLE:
                raise ImportError("Optional loading of the CUDA module is broken")
            else:
                continue

    # test the slice scatter function
    for backend in slice_backends.keys():
        image1 = slice_backends[backend](
            x=coordinates_periodic[:, 0],
            y=coordinates_periodic[:, 1],
            z=coordinates_periodic[:, 2],
            m=masses_periodic,
            h=hsml_periodic,
            z_slice=0.5,
            xres=pixel_resolution,
            yres=pixel_resolution,
            box_x=boxsize,
            box_y=boxsize,
            box_z=boxsize,
        )
        image2 = slice_backends[backend](
            x=coordinates_non_periodic[:, 0],
            y=coordinates_non_periodic[:, 1],
            z=coordinates_non_periodic[:, 2],
            m=masses_non_periodic,
            h=hsml_non_periodic,
            z_slice=0.5,
            xres=pixel_resolution,
            yres=pixel_resolution,
            box_x=0.0,
            box_y=0.0,
            box_z=0.0,
        )

        assert (image1 == image2).all()

    # test the volume rendering scatter function
    scatter = volume_render_backends["scatter"]
    image1 = scatter(
        x=coordinates_periodic[:, 0],
        y=coordinates_periodic[:, 1],
        z=coordinates_periodic[:, 2],
        m=masses_periodic,
        h=hsml_periodic,
        res=voxel_resolution,
        box_x=boxsize,
        box_y=boxsize,
        box_z=boxsize,
    )
    image2 = scatter(
        x=coordinates_non_periodic[:, 0],
        y=coordinates_non_periodic[:, 1],
        z=coordinates_non_periodic[:, 2],
        m=masses_non_periodic,
        h=hsml_non_periodic,
        res=voxel_resolution,
        box_x=0.0,
        box_y=0.0,
        box_z=0.0,
    )

    assert (image1 == image2).all()


def test_volume_render_and_unfolded_deposit():
    """
    Test that volume render and unfolded deposit can give
    the same result.
    """

    x = np.array([100, 200])
    y = np.array([100, 200])
    z = np.array([100, 200])
    m = np.array([1, 1])
    h = np.array([1e-10, 1e-10])

    res = 256
    boxsize = 1.0 * res

    # 1.0 implies no folding
    deposition = deposit(x, y, z, m, res, 1.0, boxsize, boxsize, boxsize)

    # Need to norm coords and box for the volume render
    volume = volume_render_backends["scatter"](
        x=x / boxsize,
        y=y / boxsize,
        z=z / boxsize,
        m=m,
        h=h / boxsize,
        res=res,
        box_x=boxsize,
        box_y=boxsize,
        box_z=boxsize,
    )

    assert np.allclose(deposition, volume.view(np.ndarray))


def test_folding_deposit():
    """
    Tests that the deposit returns the 'correct' units.
    """

    x = np.array([100, 200])
    y = np.array([100, 200])
    z = np.array([100, 200])
    m = np.array([1, 1])

    res = 256
    boxsize = 1.0 * res

    # 1.0 implies no folding
    deposition_1 = deposit(x, y, z, m, res, 1.0, boxsize, boxsize, boxsize)

    # 2.0 implies folding by factor of 8
    deposition_2 = deposit(x, y, z, m, res, 2.0, boxsize, boxsize, boxsize)

    assert deposition_1[100, 100, 100] * 8.0 == deposition_2[200, 200, 200]


@requires("cosmological_volume.hdf5")
def test_volume_render_and_unfolded_deposit_with_units(filename):
    data = load(filename)
    data.gas.smoothing_lengths = 1e-30 * data.gas.smoothing_lengths
    npix = 64

    # Deposit the particles
    deposition = render_to_deposit(
        data.gas, npix, project="masses", folding=0, parallel=False
    ).to_physical()

    # Volume render the particles
    volume = render_gas(data, npix, parallel=False).to_physical()

    mean_density_deposit = (np.sum(deposition) / npix ** 3).to("Msun / kpc**3").v
    mean_density_volume = (np.sum(volume) / npix ** 3).to("Msun / kpc**3").v
    mean_density_calculated = (
        (np.sum(data.gas.masses) / (data.metadata.boxsize[0] * data.metadata.a) ** 3)
        .to("Msun / kpc**3")
        .v
    )

    assert np.isclose(mean_density_deposit, mean_density_calculated)
    assert np.isclose(mean_density_volume, mean_density_calculated, rtol=0.2)
    assert np.isclose(mean_density_deposit, mean_density_volume, rtol=0.2)


@requires("cosmo_volume_example.hdf5")
def test_dark_matter_power_spectrum(filename, save=False):
    data = load(filename)

    data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
        data.dark_matter.coordinates, data.metadata.boxsize, kernel_gamma=1.8
    )

    # Collate a bunch of raw depositions
    folds = {}

    min_k = cosmo_quantity(
        1e-2,
        unyt.Mpc ** -1,
        comoving=True,
        cosmo_factor=cosmo_factor(a ** -1, data.metadata.scale_factor),
    )
    max_k = cosmo_quantity(
        1e2,
        unyt.Mpc ** -1,
        comoving=True,
        cosmo_factor=cosmo_factor(a ** -1, data.metadata.scale_factor),
    )

    bins = np.geomspace(min_k, max_k, 32)

    output = {}
    for npix in [32, 128]:
        # Deposit the particles
        deposition = render_to_deposit(
            data.dark_matter, npix, project="masses", folding=0, parallel=False
        ).to("Msun / Mpc**3")

        # Calculate the power spectrum
        k, power_spectrum, scatter = deposition_to_power_spectrum(
            deposition, data.metadata.boxsize, folding=0, wavenumber_bins=bins
        )

        if npix == 32:
            folds[0] = deposition

        output[npix] = (k, power_spectrum, scatter)

    folding_output = {}

    for folding in [2, 4, 6, 8]:  # , 8.0, 512.0]:
        # Deposit the particles
        deposition = render_to_deposit(
            data.dark_matter, 32, project="masses", folding=folding, parallel=False
        ).to("Msun / Mpc**3")

        # Calculate the power spectrum
        k, power_spectrum, scatter = deposition_to_power_spectrum(
            deposition, data.metadata.boxsize, folding=folding, wavenumber_bins=bins
        )

        folds[folding] = deposition

        folding_output[2 ** folding] = (k, power_spectrum, scatter)

    # Now try doing them all together at once.

    _, all_centers, all_ps, folding_tracker = folded_depositions_to_power_spectrum(
        depositions=folds,
        box_size=data.metadata.boxsize,
        number_of_wavenumber_bins=32,
        cross_depositions=None,
        wavenumber_range=(min_k, max_k),
        log_wavenumber_bins=True,
    )

    # import pdb; pdb.set_trace()

    if save:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.cm import ScalarMappable

        with unyt.matplotlib_support:
            for npix, (k, power_spectrum, _) in output.items():
                plt.plot(k, power_spectrum, label=f"Npix {npix}")

            for fold_id, (k, power_spectrum, _) in folding_output.items():
                plt.plot(
                    k, power_spectrum, label=f"Fold {fold_id} (Npix 32)", ls="dotted"
                )

            cmap = plt.get_cmap()
            norm = LogNorm()
            colors = cmap(norm(folding_tracker))
            plt.scatter(
                all_centers, all_ps, label="Full Fold", color=colors, edgecolor="pink"
            )
            plt.colorbar(
                mappable=ScalarMappable(norm=norm, cmap=cmap),
                ax=plt.gca(),
                label="Folding",
            )

            plt.loglog()
            plt.axvline(
                2 * np.pi / data.metadata.boxsize[0].to("Mpc"),
                color="black",
                linestyle="--",
            )
            plt.axvline(
                2
                * np.pi
                / (
                    data.metadata.boxsize[0].to("Mpc")
                    / len(data.dark_matter.smoothing_lengths) ** (1 / 3)
                ),
                color="black",
                linestyle="--",
            )
            plt.text(0.05, 0.05, data.metadata.boxsize, transform=plt.gca().transAxes)
            plt.legend()
            plt.savefig("dark_matter_power_spectrum.png")

    return
