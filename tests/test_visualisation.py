import pytest
from swiftsimio import load
from swiftsimio.visualisation import scatter, slice, volume_render
from swiftsimio.visualisation.projection import (
    scatter_parallel,
    project_gas,
    project_pixel_grid,
)
from swiftsimio.visualisation.slice import (
    slice_scatter,
    slice_scatter_parallel,
    slice_gas,
)
from swiftsimio.visualisation.volume_render import render_gas
from swiftsimio.visualisation.volume_render import scatter as volume_scatter
from swiftsimio.visualisation.power_spectrum import (
    deposit,
    deposition_to_power_spectrum,
    render_to_deposit,
    folded_depositions_to_power_spectrum,
)
from swiftsimio.visualisation.projection_backends import backends, backends_parallel
from swiftsimio.visualisation.smoothing_length_generation import (
    generate_smoothing_lengths,
)
from swiftsimio.optional_packages import CudaSupportError, CUDA_AVAILABLE
from swiftsimio.objects import cosmo_array, a
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

    for backend in backends.keys():
        try:
            image = backends[backend](
                np.array([0.0, 1.0, 1.0, -0.000001]),
                np.array([0.0, 0.0, 1.0, 1.000001]),
                np.array([1.0, 1.0, 1.0, 1.0]),
                np.array([0.2, 0.2, 0.2, 0.000002]),
                256,
                1.0,
                1.0,
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
    np.random.seed(971263)
    # Width of 0.8 centered on 0.5, 0.5.
    x = 0.8 * np.random.rand(100) + 0.1
    y = 0.8 * np.random.rand(100) + 0.1
    m = np.ones_like(x)
    h = 0.05 * np.ones_like(x)

    resolutions = [8, 16, 32, 64, 128, 256, 512]
    total_mass = np.sum(m)

    for resolution in resolutions:
        image = scatter(x, y, m, h, resolution, 1.0, 1.0)
        mass_in_image = image.sum() / (resolution**2)

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

    image = scatter(coordinates[0], coordinates[1], masses, hsml, resolution, 1.0, 1.0)
    image_par = scatter_parallel(
        coordinates[0], coordinates[1], masses, hsml, resolution, 1.0, 1.0
    )

    if save:
        imsave("test_image_creation.png", image)

    assert np.isclose(image, image_par).all()

    return


def test_slice(save=False):
    image = slice(
        np.array([0.0, 1.0, 1.0, -0.000001]),
        np.array([0.0, 0.0, 1.0, 1.000001]),
        np.array([0.0, 0.0, 1.0, 1.000001]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.2, 0.2, 0.2, 0.000002]),
        0.99,
        256,
        1.0,
        1.0,
        1.0,
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
        1.0,
        1.0,
        1.0,
    )
    image_par = slice_scatter_parallel(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        masses,
        hsml,
        z_slice,
        resolution,
        1.0,
        1.0,
        1.0,
    )

    if save:
        imsave("test_image_creation.png", image)

    assert np.isclose(image, image_par).all()

    return


def test_volume_render():
    # render image
    volume_render.scatter(
        np.array([0.0, 1.0, 1.0, -0.000001]),
        np.array([0.0, 0.0, 1.0, 1.000001]),
        np.array([0.0, 0.0, 1.0, 1.000001]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.2, 0.2, 0.2, 0.000002]),
        64,
        1.0,
        1.0,
        1.0,
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
        coordinates[0],
        coordinates[1],
        coordinates[2],
        masses,
        hsml,
        resolution,
        1.0,
        1.0,
        1.0,
    )
    image_par = volume_render.scatter_parallel(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        masses,
        hsml,
        resolution,
        1.0,
        1.0,
        1.0,
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
        data, 256, parallel=True, region=[0 * bs, 0.00 * bs, 0.25 * bs, 0.75 * bs]
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
    backends["histogram"](x, y, m, h, resolution, 1.0, 1.0)

    for backend in backends_parallel.keys():
        try:
            backends[backend](x, y, m, h, resolution, 1.0, 1.0)
        except CudaSupportError:
            if CUDA_AVAILABLE:
                raise ImportError("Optional loading of the CUDA module is broken")
            else:
                continue

    slice_scatter_parallel(x, y, z, m, h, 0.2, resolution, 1.0, 1.0, 1.0)

    volume_render.scatter_parallel(x, y, z, m, h, resolution, 1.0, 1.0, 1.0)


@requires("cosmological_volume.hdf5")
def test_comoving_versus_physical(filename):
    """
    Test what happens if you try to mix up physical and comoving quantities.
    """

    for func, aexp in [(project_gas, -2.0), (slice_gas, -3.0), (render_gas, -3.0)]:
        # normal case: everything comoving
        data = load(filename)
        # we force the default (project="masses") to check the cosmo_factor
        # conversion in this case
        img = func(data, resolution=256, project=None)
        assert img.comoving
        assert img.cosmo_factor.expr == a**aexp
        img = func(data, resolution=256, project="densities")
        assert img.comoving
        assert img.cosmo_factor.expr == a ** (aexp - 3.0)
        # try to mix comoving coordinates with a physical variable
        data.gas.densities.convert_to_physical()
        with pytest.raises(AttributeError, match="not compatible with comoving"):
            img = func(data, resolution=256, project="densities")
        # convert coordinates to physical (but not smoothing lengths)
        data.gas.coordinates.convert_to_physical()
        with pytest.raises(AttributeError, match=""):
            img = func(data, resolution=256, project="masses")
        # also convert smoothing lengths to physical
        data.gas.smoothing_lengths.convert_to_physical()
        # masses are always compatible with either
        img = func(data, resolution=256, project="masses")
        # check that we get a physical result
        assert not img.comoving
        assert img.cosmo_factor.expr == a**aexp
        # densities are still compatible with physical
        img = func(data, resolution=256, project="densities")
        assert not img.comoving
        assert img.cosmo_factor.expr == a ** (aexp - 3.0)
        # now try again with comoving densities
        data.gas.densities.convert_to_comoving()
        with pytest.raises(AttributeError, match="not compatible with physical"):
            img = func(data, resolution=256, project="densities")


@requires("cosmological_volume.hdf5")
def test_nongas_smoothing_lengths(filename):
    """
    Test that the visualisation tools to calculate smoothing lengths give usable results.
    """

    # If project_pixel_grid runs without error the smoothing lengths seem usable.
    data = load(filename)
    data.dark_matter.smoothing_length = generate_smoothing_lengths(
        data.dark_matter.coordinates, data.metadata.boxsize, kernel_gamma=1.8
    )
    project_pixel_grid(
        data.dark_matter,
        boxsize=data.metadata.boxsize,
        resolution=256,
        project="masses",
    )
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
    for backend in backends.keys():
        try:
            image1 = backends[backend](
                x=coordinates_periodic[:, 0],
                y=coordinates_periodic[:, 1],
                m=masses_periodic,
                h=hsml_periodic,
                res=pixel_resolution,
                box_x=boxsize,
                box_y=boxsize,
            )
            image2 = backends[backend](
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
    image1 = slice_scatter(
        x=coordinates_periodic[:, 0],
        y=coordinates_periodic[:, 1],
        z=coordinates_periodic[:, 2],
        m=masses_periodic,
        h=hsml_periodic,
        z_slice=0.5,
        res=pixel_resolution,
        box_x=boxsize,
        box_y=boxsize,
        box_z=boxsize,
    )
    image2 = slice_scatter(
        x=coordinates_non_periodic[:, 0],
        y=coordinates_non_periodic[:, 1],
        z=coordinates_non_periodic[:, 2],
        m=masses_non_periodic,
        h=hsml_non_periodic,
        z_slice=0.5,
        res=pixel_resolution,
        box_x=0.0,
        box_y=0.0,
        box_z=0.0,
    )

    assert (image1 == image2).all()

    # test the volume rendering scatter function
    image1 = volume_render.scatter(
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

    image2 = volume_render.scatter(
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

    # Need to norm for the volume render
    volume = volume_scatter(
        x / boxsize,
        y / boxsize,
        z / boxsize,
        m,
        h / boxsize,
        res,
        boxsize,
        boxsize,
        boxsize,
    )

    assert np.allclose(deposition, volume)


def test_folding_deposit():
    """
    Tests that the deposit returns the 'correct' units.
    """

    x = np.array([100, 200])
    y = np.array([100, 200])
    z = np.array([100, 200])
    m = np.array([1, 1])
    h = np.array([1e-10, 1e-10])

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

    mean_density_deposit = (np.sum(deposition) / npix**3).to("Msun / kpc**3").v
    mean_density_volume = (np.sum(volume) / npix**3).to("Msun / kpc**3").v
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

    min_k = 1e-2 / unyt.Mpc
    max_k = 1e2 / unyt.Mpc

    bins = unyt.unyt_array(
        np.logspace(np.log10(min_k.v), np.log10(max_k.v), 32),
        units=min_k.units,
    )

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

        folding_output[2**folding] = (k, power_spectrum, scatter)

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
                all_centers,
                all_ps,
                label="Full Fold",
                color=colors,
                edgecolor="pink",
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
