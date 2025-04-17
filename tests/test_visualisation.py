import pytest
from swiftsimio import load, mask
from swiftsimio.visualisation.projection import project_gas, project_pixel_grid
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.volume_render import render_gas

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

from swiftsimio.objects import cosmo_array, cosmo_quantity, a
from unyt.array import unyt_array
import unyt

import numpy as np


try:
    from matplotlib.pyplot import imsave
except ImportError:
    pass


def fraction_within_tolerance(a, b, frac=0.99, tol=0.1):
    """
    Compare array values in ``a`` and ``b``, return ``True`` if a fraction
    ``frac`` of values are within a retlative tolerance ``tol`` of their
    counterparts.

    Paramters
    ---------
    a : ~swiftsimio.objects.cosmo_array
        The first array to compare.
    b : ~swiftsimio.objects.cosmo_array
        The second array to compare
    frac : float
        The fraction of the values that must be within the tolerance.
    tol : float
        The relative tolerance for matching.

    Returns
    -------
    out : bool
        ``True`` if enough values match within the tolerance, ``False`` otherwise.
    """
    assert a.shape == b.shape

    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "divide by zero encountered in divide")
        sup.filter(RuntimeWarning, "invalid value encountered in divide")
        ratios = np.abs((a / b).to_value(unyt.dimensionless) - 1)
    ratios[np.isnan(ratios)] = 0  # 0 == 0 is a match
    return np.sum(ratios < tol) / a.size > frac


class TestProjection:
    @pytest.mark.parametrize("backend", projection_backends.keys())
    def test_scatter(self, backend, save=False):
        """
        Tests the scatter functions from all backends.
        """
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")
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
                pytest.skip("CUDA is not available")

        if save:
            imsave("test_image_creation.png", image)

        return

    def test_scatter_mass_conservation(self):
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

    def test_scatter_parallel(self, save=False):
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

    @pytest.mark.parametrize("backend", projection_backends.keys())
    def test_equivalent_regions(self, backend, cosmo_volume_example):
        """
        Here we test that various regions are (close enough to) equivalent.
        The ref_img is just a projection through the whole box.
        The big_img is the box tiled 3x3 times, confirming that we can periodically wrap
        as many times as we like.
        The far_img is way, way outside the box, confirming that we can place the region
        anywhere.
        The depth_img is a reference image with limited projection depth in the box.
        The neg_depth_img is as the depth_img but with the z range in negative values.
        The wrap_depth_img is as the depth_img but with the z range beyond the box length.
        The straddled_depth_img compares to the ref_img - it projects through the whole
        box but straddling z=0 instead of starting there.
        The edge_img is the only non-periodic case, framed to only partially contain the
        box. We check that it matches the expected region of the ref_img (with the edges
        trimmed a bit).
        """
        sd = load(cosmo_volume_example)
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")
        parallel = True
        lbox = sd.metadata.boxsize[0].to_comoving().to_value(unyt.Mpc)
        box_res = 256
        ref_img = project_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            periodic=True,
            backend=backend,
        )
        big_img = project_gas(
            sd,
            region=cosmo_array(
                [0, 3 * lbox, 0, 3 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res * 3,
            parallel=parallel,
            periodic=True,
            backend=backend,
        )
        far_img = project_gas(
            sd,
            region=cosmo_array(
                [50 * lbox, 51 * lbox, 50 * lbox, 51 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            periodic=True,
            backend=backend,
        )
        depth_img = project_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox, 0, 0.3 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            backend=backend,
        )
        neg_depth_img = project_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox, -lbox, -0.7 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            backend=backend,
        )
        wrap_depth_img = project_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox, lbox, 1.3 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            backend=backend,
        )
        straddled_depth_img = project_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox, -0.5 * lbox, 0.5 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            backend=backend,
        )
        edge_img = project_gas(
            sd,
            region=cosmo_array(
                [-0.25 * lbox, 0.75 * lbox, 0.5 * lbox, 1.5 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            periodic=False,
            backend=backend,
        )
        edge_mask = np.s_[box_res // 6 : -box_res // 6, box_res // 6 : -box_res // 6]
        assert fraction_within_tolerance(
            edge_img[box_res // 4 :, : box_res // 2][edge_mask],
            ref_img[: 3 * box_res // 4, box_res // 2 :][edge_mask],
        )
        assert np.allclose(far_img, ref_img)
        assert fraction_within_tolerance(
            big_img, np.concatenate([np.hstack([ref_img] * 3)] * 3, axis=1)
        )
        assert np.allclose(depth_img, neg_depth_img)
        assert np.allclose(depth_img, wrap_depth_img)
        assert np.allclose(ref_img, straddled_depth_img)

    @pytest.mark.parametrize("backend", projection_backends.keys())
    def test_periodic_boundary_wrapping(self, backend):
        """
        Test that periodic boundary wrapping works.
        """
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")

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
                pytest.skip("CUDA is not available")


class TestSlice:
    def test_slice(self, save=False):
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

    def test_slice_parallel(self, save=False):
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

    @pytest.mark.parametrize("backend", slice_backends.keys())
    def test_periodic_boundary_wrapping(self, backend):
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")

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

        # test the slice scatter function
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

    @pytest.mark.parametrize("backend", slice_backends.keys())
    def test_equivalent_regions(self, backend, cosmo_volume_example):
        """
        Here we test that various regions are (close enough to) equivalent.
        The ref_img is just a slice through the whole box at z=0.5 * lbox.
        The big_img is the box tiled 3x3 times, confirming that we can periodically wrap
        as many times as we like.
        The far_img is way, way outside the box, confirming that we can place the region
        anywhere.
        The neg_img places the slice at z=-0.5 * lbox.
        The wrap_img places the slice at z=1.5 * lbox.
        The edge_img is the only non-periodic case, framed to only partially contain the
        box. We check that it matches the expected region of the ref_img (with the edges
        trimmed a bit).
        """
        sd = load(cosmo_volume_example)
        parallel = True
        lbox = sd.metadata.boxsize[0].to_comoving().to_value(unyt.Mpc)
        box_res = 256
        ref_img = slice_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            z_slice=cosmo_quantity(
                0.5 * lbox,
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            periodic=True,
            backend=backend,
        )
        big_img = slice_gas(
            sd,
            region=cosmo_array(
                [0, 3 * lbox, 0, 3 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            z_slice=cosmo_quantity(
                0.5 * lbox,
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res * 3,
            parallel=parallel,
            periodic=True,
            backend=backend,
        )
        far_img = slice_gas(
            sd,
            region=cosmo_array(
                [50 * lbox, 51 * lbox, 50 * lbox, 51 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            z_slice=cosmo_quantity(
                0.5 * lbox,
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            periodic=True,
            backend=backend,
        )
        neg_img = slice_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox, -lbox, -0.7 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            z_slice=cosmo_quantity(
                -0.5 * lbox,
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            backend=backend,
        )
        wrap_img = slice_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox, lbox, 1.3 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            z_slice=cosmo_quantity(
                1.5 * lbox,
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            backend=backend,
        )
        edge_img = slice_gas(
            sd,
            region=cosmo_array(
                [-0.25 * lbox, 0.75 * lbox, 0.5 * lbox, 1.5 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            z_slice=cosmo_quantity(
                0.5 * lbox,
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            periodic=False,
            backend=backend,
        )
        edge_mask = np.s_[box_res // 6 : -box_res // 6, box_res // 6 : -box_res // 6]
        assert fraction_within_tolerance(
            edge_img[box_res // 4 :, : box_res // 2][edge_mask],
            ref_img[: 3 * box_res // 4, box_res // 2 :][edge_mask],
            frac={"nearest_neighbours": 0.8}.get(backend, 0.99),
        )
        assert np.allclose(far_img, ref_img)
        assert fraction_within_tolerance(
            big_img,
            np.concatenate([np.hstack([ref_img] * 3)] * 3, axis=1),
            frac={"nearest_neighbours": 0.8}.get(backend, 0.99),
        )
        assert np.allclose(neg_img, ref_img)
        assert np.allclose(wrap_img, ref_img)


class TestVolumeRender:
    def test_volume_render(self):
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

    def test_volume_parallel(self):
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

    def test_volume_render_and_unfolded_deposit(self):
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

    def test_folding_deposit(self):
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

    def test_volume_render_and_unfolded_deposit_with_units(self, cosmological_volume):
        data = load(cosmological_volume)
        data.gas.smoothing_lengths = 1e-30 * data.gas.smoothing_lengths
        npix = 64

        # Deposit the particles
        deposition = render_to_deposit(
            data.gas, npix, project="masses", folding=0, parallel=False
        ).to_physical()

        # Volume render the particles
        volume = render_gas(data, npix, parallel=False).to_physical()

        mean_density_deposit = (
            (np.sum(deposition) / npix ** 3)
            .to_comoving()
            .to_value(unyt.solMass / unyt.kpc ** 3)
        )
        mean_density_volume = (
            (np.sum(volume) / npix ** 3)
            .to_comoving()
            .to_value(unyt.solMass / unyt.kpc ** 3)
        )
        mean_density_calculated = (
            (np.sum(data.gas.masses) / np.prod(data.metadata.boxsize))
            .to_comoving()
            .to_value(unyt.solMass / unyt.kpc ** 3)
        )

        assert np.isclose(mean_density_deposit, mean_density_calculated)
        assert np.isclose(mean_density_volume, mean_density_calculated, rtol=0.2)
        assert np.isclose(mean_density_deposit, mean_density_volume, rtol=0.2)

    def test_periodic_boundary_wrapping(self):

        voxel_resolution = 10
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

    def test_equivalent_regions(self, cosmo_volume_example):
        """
        Here we test that various regions are (close enough to) equivalent.
        The ref_img is just a render of the whole box.
        The big_img is the box tiled 3x3x3 times, confirming that we can periodically wrap
        as many times as we like.
        The far_img is way, way outside the box, confirming that we can place the region
        anywhere.
        The straddled_img is offset by half a box length, the two halves of the result
        should agree with the opposite halves of the ref_img.
        The edge_img is the only non-periodic case, framed to only partially contain the
        box. We check that it matches the expected region of the ref_img (with the edges
        trimmed a bit).
        """
        sd = load(cosmo_volume_example)
        parallel = False  # memory gets a bit out of hand otherwise
        lbox = sd.metadata.boxsize[0].to_comoving().to_value(unyt.Mpc)
        box_res = 32
        ref_img = render_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox, 0, lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            periodic=True,
        )
        big_img = render_gas(
            sd,
            region=cosmo_array(
                [0, 3 * lbox, 0, 3 * lbox, 0, 3 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res * 3,
            parallel=parallel,
            periodic=True,
        )
        far_img = render_gas(
            sd,
            region=cosmo_array(
                [50 * lbox, 51 * lbox, 50 * lbox, 51 * lbox, 50 * lbox, 51 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            periodic=True,
        )
        straddled_img = render_gas(
            sd,
            region=cosmo_array(
                [0, lbox, 0, lbox, -0.5 * lbox, 0.5 * lbox],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
        )
        edge_img = render_gas(
            sd,
            region=cosmo_array(
                [
                    -0.25 * lbox,
                    0.75 * lbox,
                    0.5 * lbox,
                    1.5 * lbox,
                    0.5 * lbox,
                    1.5 * lbox,
                ],
                unyt.Mpc,
                comoving=True,
                scale_factor=sd.metadata.a,
                scale_exponent=1,
            ),
            resolution=box_res,
            parallel=parallel,
            periodic=False,
        )
        edge_mask = np.s_[
            box_res // 6 : -box_res // 6,
            box_res // 6 : -box_res // 6,
            box_res // 6 : -box_res // 6,
        ]
        assert fraction_within_tolerance(
            edge_img[box_res // 4 :, : box_res // 2, : box_res // 2][edge_mask],
            ref_img[: 3 * box_res // 4, box_res // 2 :, box_res // 2 :][edge_mask],
        )
        assert np.allclose(far_img, ref_img)
        slab = np.concatenate([np.hstack([ref_img] * 3)] * 3, axis=1)
        cube = np.concatenate([slab] * 3, axis=2)
        assert fraction_within_tolerance(big_img, cube)
        assert fraction_within_tolerance(
            ref_img,
            np.concatenate(
                (
                    straddled_img[..., box_res // 2 :],
                    straddled_img[..., : box_res // 2],
                ),
                axis=-1,
            ),
        )


def test_selection_render(cosmological_volume):
    data = load(cosmological_volume)
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


def test_comoving_versus_physical(cosmological_volume):
    """
    Test what happens if you try to mix up physical and comoving quantities.
    """

    # this test is pretty slow if we don't mask out some particles
    m = mask(cosmological_volume)
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
        data = load(cosmological_volume, mask=m)
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


def test_nongas_smoothing_lengths(cosmological_volume):
    """
    Test that the visualisation tools to calculate smoothing lengths give usable results.
    """

    # If project_gas runs without error the smoothing lengths seem usable.
    data = load(cosmological_volume)
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


class TestPowerSpectrum:
    def test_dark_matter_power_spectrum(self, cosmo_volume_example, save=False):
        data = load(cosmo_volume_example)

        data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
            data.dark_matter.coordinates, data.metadata.boxsize, kernel_gamma=1.8
        )

        # Collate a bunch of raw depositions
        folds = {}

        min_k = cosmo_quantity(
            1e-2,
            unyt.Mpc ** -1,
            comoving=True,
            scale_factor=data.metadata.scale_factor,
            scale_exponent=-1,
        )
        max_k = cosmo_quantity(
            1e2,
            unyt.Mpc ** -1,
            comoving=True,
            scale_factor=data.metadata.scale_factor,
            scale_exponent=-1,
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
            boxsize=data.metadata.boxsize,
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
                        k,
                        power_spectrum,
                        label=f"Fold {fold_id} (Npix 32)",
                        ls="dotted",
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
                plt.text(
                    0.05, 0.05, data.metadata.boxsize, transform=plt.gca().transAxes
                )
                plt.legend()
                plt.savefig("dark_matter_power_spectrum.png")

        return
