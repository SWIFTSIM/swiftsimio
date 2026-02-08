"""Tests of the visualisation tools."""

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


def fraction_within_tolerance(
    a: np.ndarray, b: np.ndarray, frac: float = 1.0, tol: float = 1e-3
) -> bool:
    """
    Check that two arrays have most values within a desired tolerance.

    Compare array values in ``a`` and ``b``, return ``True`` if a fraction
    ``frac`` of values are within a retlative tolerance ``tol`` of their
    counterparts.

    Parameters
    ----------
    a : ~swiftsimio.objects.cosmo_array
        The first array to compare.

    b : ~swiftsimio.objects.cosmo_array
        The second array to compare.

    frac : float
        The fraction of the values that must be within the tolerance.

    tol : float
        The relative tolerance for matching.

    Returns
    -------
    bool
        ``True`` if enough values match within the tolerance, ``False`` otherwise.
    """
    assert a.shape == b.shape

    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "divide by zero encountered in divide")
        sup.filter(RuntimeWarning, "invalid value encountered in divide")
        ratios = np.abs((a / b).to_value(unyt.dimensionless) - 1)
    ratios[np.isnan(ratios)] = 0  # 0 == 0 is a match
    return np.sum(ratios < tol) / a.size >= frac


class TestProjection:
    """Tests for the projection functions in the visualisation tools."""

    @pytest.mark.parametrize("backend", projection_backends.keys())
    def test_scatter(self, backend, save=False):
        """Tests the scatter functions from all projection backends."""
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
        """Check that projecting insets all of the mass into the projection."""
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
            mass_in_image = image.sum() / (resolution**2)

            # Check mass conservation to 5%
            assert np.isclose(mass_in_image.view(np.ndarray), total_mass, 0.05)

        return

    def test_scatter_parallel(self, save=False):
        """Check that parallel projection is equivalent to serial projection."""
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
    def test_equivalent_regions(self, backend, cosmological_volume_only_single):
        """
        Check that equivalent regions are (close enough to) equivalent.

        + The ref_img is just a projection through the whole box.
        + The big_img is the box tiled 3x3 times, confirming that we can periodically wrap
          as many times as we like.
        + The far_img is way, way outside the box, confirming that we can place the region
          anywhere.
        + The depth_img is a reference image with limited projection depth in the box.
        + The neg_depth_img is as the depth_img but with the z range in negative values.
        + The wrap_depth_img is as the depth_img but with the z range beyond the box
          length.
        + The straddled_depth_img compares to the ref_img - it projects through the whole
          box but straddling z=0 instead of starting there.
        + The edge_img is the only non-periodic case, framed to only partially contain the
          box. We check that it matches the expected region of the ref_img (with the edges
          trimmed a bit).

        It would be good to split this test into several tests, but avoid re-computing the
        ref_img for each one. Could use a fixture.
        """
        sd = load(cosmological_volume_only_single)
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")
        parallel = True
        lbox = sd.metadata.boxsize[0].to_comoving().to_value(unyt.Mpc)
        box_res = 256
        with np.errstate(
            invalid="ignore"
        ):  # invalid value encountered in divide happens sometimes
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
                    [0 * lbox, 3 * lbox, 0 * lbox, 3 * lbox],
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
        # The comparison of the periodic box projection (ref_img) and the version
        # rendering the same number of pixels per periodic box but tiling the box 3x3
        # times (big_img) is infuriatingly tricky. Tests in TestProjectionBackends show
        # that rendering works correctly at the particle level in simplified versions of
        # this test. It seems trivial that the two setups should give the same answer, but
        # it seems like differences sneak in because for the ref_img in the backend one
        # box is mapped to a normalized coordinate interval [0, 1). However for the big
        # image the same box is mapped to [0, 0.333) such that the three boxes per image
        # axis fit in [0, 1). This seems to introduce unavoidable floating point
        # arithmetic differences that lead to residuals in the images (or there's another
        # bug that hasn't been found yet). Assuming the former, I'm setting the test to
        # require 99% of pixels to agree within 1%, and adding an additional assertion
        # that there are no differences bigger than 2%. This holds even when setting
        # box_res=16 in this test (issues tend to get worse at lower pixel counts), and at
        # box_res=256 (which is what we run the test at) all pixels agree to about one
        # part in ~1e13.
        assert fraction_within_tolerance(
            big_img,
            np.concatenate([np.hstack([ref_img] * 3)] * 3, axis=1),
            frac=0.99,
            tol=0.01,
        )
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in divide")
            assert (
                np.nanmax(
                    np.abs(
                        big_img / np.concatenate([np.hstack([ref_img] * 3)] * 3, axis=1)
                    ).to_physical_value(unyt.dimensionless)
                    - 1
                )
                < 0.02
            )
        assert np.allclose(depth_img, neg_depth_img)
        assert np.allclose(depth_img, wrap_depth_img)
        assert np.allclose(ref_img, straddled_depth_img)

    @pytest.mark.parametrize("backend", projection_backends.keys())
    def test_periodic_boundary_wrapping(self, backend):
        """Test that a single particle wraps properly around the periodic boundary."""
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
    """Tests for the slice functions in the visualisation tools."""

    def test_slice(self, save=False):
        """Just run a slice and make sure we don't crash."""
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
        """Check that parallel slice and serial slice are equivalent."""
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
        """Check that a single particle wraps properly around the periodic boundary."""
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
    def test_equivalent_regions(self, backend, cosmological_volume_only_single):
        """
        Check that slices of equivalent regions are (close enough to) equivalent.

        + The ref_img is just a slice through the whole box at z=0.5 * lbox.
        + The big_img is the box tiled 3x3 times, confirming that we can periodically wrap
          as many times as we like.
        + The far_img is way, way outside the box, confirming that we can place the region
          anywhere.
        + The neg_img places the slice at z=-0.5 * lbox.
        + The wrap_img places the slice at z=1.5 * lbox.
        + The edge_img is the only non-periodic case, framed to only partially contain the
          box. We check that it matches the expected region of the ref_img (with the edges
          trimmed a bit).

        It would be good to split this test into several tests, but avoid re-computing the
        ref_img for each one. Could use a fixture.
        """
        sd = load(cosmological_volume_only_single)
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
            frac={"nearest_neighbours": 0.5}.get(backend, 1.0),
        )
        assert np.allclose(far_img, ref_img)
        assert fraction_within_tolerance(
            big_img,
            np.concatenate([np.hstack([ref_img] * 3)] * 3, axis=1),
            frac={"nearest_neighbours": 0.5}.get(backend, 1.0),
        )
        assert np.allclose(neg_img, ref_img)
        assert np.allclose(wrap_img, ref_img)


class TestVolumeRender:
    """Tests for the volume render functions in the visualisation tools."""

    def test_volume_render(self):
        """Just run a volume render and make sure we don't crash."""
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
        """Check that volume render in parallel matches serial volume render."""
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
        """Test that volume render and unfolded deposit can give the same result."""
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
        """Tests that the deposit returns the correct units."""
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

    def test_volume_render_and_unfolded_deposit_with_units(
        self, cosmological_volume_only_single
    ):
        """Check that the density in the render matches the input density."""
        data = load(cosmological_volume_only_single)
        data.gas.smoothing_lengths = 1e-30 * data.gas.smoothing_lengths
        npix = 64

        # Deposit the particles
        deposition = render_to_deposit(
            data.gas, npix, project="masses", folding=0, parallel=False
        ).to_physical()

        # Volume render the particles
        volume = render_gas(data, npix, parallel=False).to_physical()

        mean_density_deposit = (
            (np.sum(deposition) / npix**3)
            .to_comoving()
            .to_value(unyt.solMass / unyt.kpc**3)
        )
        mean_density_volume = (
            (np.sum(volume) / npix**3)
            .to_comoving()
            .to_value(unyt.solMass / unyt.kpc**3)
        )
        mean_density_calculated = (
            (np.sum(data.gas.masses) / np.prod(data.metadata.boxsize))
            .to_comoving()
            .to_value(unyt.solMass / unyt.kpc**3)
        )

        assert np.isclose(mean_density_deposit, mean_density_calculated)
        assert np.isclose(mean_density_volume, mean_density_calculated, rtol=0.2)
        assert np.isclose(mean_density_deposit, mean_density_volume, rtol=0.2)

    def test_periodic_boundary_wrapping(self):
        """Check that a single particle wraps properly around the periodic boundary."""
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

    def test_equivalent_regions(self, cosmological_volume_only_single):
        """
        Test that volume renders of equivalent regions are (close enough to) equivalent.

        + The ref_img is just a render of the whole box.
        + The big_img is the box tiled 3x3x3 times, confirming that we can periodically
          wrap as many times as we like.
        + The far_img is way, way outside the box, confirming that we can place the
          region anywhere.
        + The straddled_img is offset by half a box length, the two halves of the result
          should agree with the opposite halves of the ref_img.
        + The edge_img is the only non-periodic case, framed to only partially contain the
          box. We check that it matches the expected region of the ref_img (with the edges
          trimmed a bit).

        It would be good to split this test into several tests, but avoid re-computing the
        ref_img for each one. Could use a fixture.
        """
        sd = load(cosmological_volume_only_single)
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


def test_selection_render(cosmological_volume_only_single):
    """
    Check that we can run rendering on a sub-region of the volume.

    Just checks that nothing crashes.
    """
    data = load(cosmological_volume_only_single)
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


def test_comoving_versus_physical(cosmological_volume_only_single):
    """Test what happens if you try to mix up physical and comoving quantities."""
    # this test is pretty slow if we don't mask out some particles
    m = mask(cosmological_volume_only_single)
    boxsize = m.metadata.boxsize
    region = cosmo_array([np.zeros_like(boxsize), 0.2 * boxsize]).T
    m.constrain_spatial(region)
    for func, aexp in [(project_gas, -2.0), (slice_gas, -3.0), (render_gas, -3.0)]:
        # normal case: everything comoving
        data = load(cosmological_volume_only_single, mask=m)
        # we force the default (project="masses") to check the cosmo_factor
        # conversion in this case
        img = func(
            data,
            resolution=64,
            project="masses",
            region=region.flatten(),
            parallel=True,
        )
        assert data.gas.masses.comoving and img.comoving
        assert (img.cosmo_factor.expr - a ** (aexp)).simplify() == 0
        img = func(
            data,
            resolution=64,
            project="densities",
            region=region.flatten(),
            parallel=True,
        )
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
                img = func(
                    data,
                    resolution=64,
                    project="densities",
                    region=region.flatten(),
                    parallel=True,
                )
        assert data.gas.densities.comoving is False and img.comoving is False
        assert (img.cosmo_factor.expr - a ** (aexp - 3.0)).simplify() == 0
        # convert coordinates to physical (but not smoothing lengths):
        # the coordinates (copy) should convert back to comoving to match the masses
        data.gas.coordinates.convert_to_physical()
        with pytest.warns(UserWarning, match="Converting coordinate grid to comoving."):
            img = func(
                data,
                resolution=64,
                project="masses",
                region=region.flatten(),
                parallel=True,
            )
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
                img = func(
                    data,
                    resolution=64,
                    project="masses",
                    region=region.flatten(),
                    parallel=True,
                )
        assert data.gas.masses.comoving and img.comoving
        assert (img.cosmo_factor.expr - a**aexp).simplify() == 0
        # densities are physical, make sure this works with physical coordinates and
        # smoothing lengths
        img = func(
            data,
            resolution=64,
            project="densities",
            region=region.flatten(),
            parallel=True,
        )
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
                img = func(
                    data,
                    resolution=64,
                    project="densities",
                    region=region.flatten(),
                    parallel=True,
                )
        assert data.gas.densities.comoving and img.comoving
        assert (img.cosmo_factor.expr - a ** (aexp - 3.0)).simplify() == 0


def test_nongas_smoothing_lengths(cosmological_volume_only_single):
    """
    Check that calculating smoothing lengths give usable results.

    Tests the smoothing length generator from the visualisation tools.

    Just makes sure that we get back a unyt_array for unyt_array input, and
    a cosmo_array for cosmo_array input.
    """
    # If project_gas runs without error the smoothing lengths seem usable.
    data = load(cosmological_volume_only_single)
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
    """Tests for the visualisation module's power spectrum tools."""

    def test_dark_matter_power_spectrum(
        self, cosmological_volume_only_single, save=False
    ):
        """
        Check that power spectra can be calculated.

        Runs both a series of "raw" depositions with different fold values and
        combining them into a power spectrum. Can save a plot for inspection.
        """
        data = load(cosmological_volume_only_single)

        data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
            data.dark_matter.coordinates, data.metadata.boxsize, kernel_gamma=1.8
        )

        # Collate a bunch of raw depositions
        folds = {}

        min_k = cosmo_quantity(
            1e-2,
            unyt.Mpc**-1,
            comoving=True,
            scale_factor=data.metadata.scale_factor,
            scale_exponent=-1,
        )
        max_k = cosmo_quantity(
            1e2,
            unyt.Mpc**-1,
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

        for folding in [2, 4, 6, 8]:
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
            boxsize=data.metadata.boxsize,
            number_of_wavenumber_bins=32,
            cross_depositions=None,
            wavenumber_range=(min_k, max_k),
            log_wavenumber_bins=True,
        )

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


class TestProjectionBackends:
    """Unit tests for projection backends by inserting single or very few particles."""

    @pytest.mark.parametrize(
        "backend",
        (
            "subsampled",
            "subsampled_extreme",
            "fast",
            "renormalised",
            "gpu",
            "histogram",
        ),
    )
    def test_subsampled_no_overlap(self, backend, res=8, save=False):
        """Check a particle fully contained in a pixel."""
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")
        if save:
            import matplotlib.pyplot as plt
        # very small kernel centred in a pixel
        scatter = projection_backends[backend]
        kwargs = {
            "x": np.array([0.45], dtype=np.float64),
            "y": np.array([0.45], dtype=np.float64),
            "m": np.array([1 / res**2], dtype=np.float32),
            "h": np.array([0.01], dtype=np.float32),
            "res": res,
            "box_x": np.float64(1),
            "box_y": np.float64(1),
        }
        one_px_img = scatter(**kwargs)
        # should have one non-zero pixel
        if save:
            plt.imsave("test_image_creation.png", one_px_img.T, origin="lower")
        assert np.isclose(
            one_px_img[
                int(np.floor(kwargs["x"][0] * 10) - 1),
                int(np.floor(kwargs["y"][0] * 10) - 1),
            ],
            1.0,
        )
        assert (one_px_img > 0).sum() == 1

    @pytest.mark.parametrize("backend", ("subsampled", "subsampled_extreme", "gpu"))
    def test_subsampled_bisected_unresolved(self, backend, res=8, save=False):
        """Check a particle bisected twice into 4 pixels but smaller than the pixels."""
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")
        if save:
            import matplotlib.pyplot as plt
        # kernel bisected in both directions, smaller than pixel
        scatter = projection_backends[backend]
        kwargs = {
            "x": np.array([0.5], dtype=np.float64),
            "y": np.array([0.5], dtype=np.float64),
            "m": np.array([1 / res**2], dtype=np.float32),
            "h": np.array([0.02], dtype=np.float32),
            "res": res,
            "box_x": np.float64(1),
            "box_y": np.float64(1),
        }
        four_px_img = scatter(**kwargs)
        # should have 4 non-zero pixels
        if save:
            plt.imsave("test_image_creation.png", four_px_img.T, origin="lower")
        assert np.allclose(
            four_px_img[
                int(np.floor(kwargs["x"][0] * 10) - 2) : int(
                    np.floor(kwargs["x"][0] * 10)
                ),
                int(np.floor(kwargs["y"][0] * 10) - 2) : int(
                    np.floor(kwargs["y"][0] * 10)
                ),
            ],
            np.array([[0.25, 0.25], [0.25, 0.25]]),
            atol=0.0001,
        )
        assert (four_px_img > 0).sum() == 4

    @pytest.mark.parametrize(
        "backend", ("subsampled", "subsampled_extreme", "fast", "renormalised", "gpu")
    )
    def test_subsampled_bisected_resolved(self, backend, res=8, save=False):
        """Check a particle bisected twice and resolved by many pixels."""
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")
        if save:
            import matplotlib.pyplot as plt
        # kernel bisected in both directions, touches edge of box
        scatter = projection_backends[backend]
        kwargs = {
            "x": np.array([0.5], dtype=np.float64),
            "y": np.array([0.5], dtype=np.float64),
            "m": np.array([1 / res**2], dtype=np.float32),
            "h": np.array([0.2], dtype=np.float32),
            "res": res,
            "box_x": np.float64(1),
            "box_y": np.float64(1),
        }
        smooth_img = scatter(**kwargs)
        # should have 36 (subsampled) or 44 (subsampled_extreme) non-zero pixels
        if save:
            plt.imsave("test_image_creation.png", smooth_img.T, origin="lower")
        assert (smooth_img > 0).sum() == {
            "subsampled": 36,
            "subsampled_extreme": 44,
            "fast": 32,
            "renormalised": 32,
        }[backend]
        assert np.isclose(smooth_img.sum(), 1.0, atol=1e-2)

    @pytest.mark.parametrize("backend", ("subsampled", "subsampled_extreme", "gpu"))
    def test_subsampled_corner_pixels(self, backend, res=8, save=False):
        """Check a particle that "touches" pixels in all 4 corners."""
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")
        if save:
            import matplotlib.pyplot as plt
        # kernel in upper right corner, wraps to all 4 corners
        scatter = projection_backends[backend]
        kwargs = {
            "x": np.array([0.98], dtype=np.float64),
            "y": np.array([0.98], dtype=np.float64),
            "m": np.array([1 / res**2], dtype=np.float32),
            "h": np.array([0.03], dtype=np.float32),
            "res": res,
            "box_x": np.float64(1),
            "box_y": np.float64(1),
        }
        corner_img = scatter(**kwargs)
        # should have 4 corners non-zero
        if save:
            plt.imsave("test_image_creation.png", corner_img.T, origin="lower")
        assert corner_img[0, 0] > 0
        assert corner_img[-1, 0] > 0
        assert corner_img[0, -1] > 0
        assert corner_img[-1, -1] > 0
        assert (corner_img > 0).sum() == 4

    @pytest.mark.parametrize(
        "backend", ("subsampled", "subsampled_extreme", "fast", "renormalised", "gpu")
    )
    def test_subsampled_tiled_box(self, backend, res=8, save=True):
        """
        Check agreement between a box image and an image tiling the box.

        This time we use more than one particle to check wrapping behaviour for particles
        in different cases (resolved, unresolved, split across pixels, or not). Individual
        cases are covered in other tests, this test is intended to catch errors such as
        unintentionally writing to negative image offsets (img[-1, -1]) that can arise
        when multiple box lengths are covered in the region.
        """
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")
        if save:
            import matplotlib.pyplot as plt
        # image repeats in each quadrant with periodic wrapping
        from swiftsimio.visualisation.projection_backends.kernels import kernel_gamma

        scatter = projection_backends[backend]
        pos = np.array([0.49, 0.27, 0.25, 0.25], dtype=np.float64)
        h = np.array([0.1, 0.001, 0.001, 0.1], dtype=np.float32) / kernel_gamma
        kwargs = {
            "x": pos,
            "y": pos,
            "m": np.array([1 / res**2] * pos.size, dtype=np.float32),
            "h": h,
            "res": res,
            "box_x": np.float64(0.5),
            "box_y": np.float64(0.5),
        }
        ref_kwargs = {
            "x": pos * 2,
            "y": pos * 2,
            "m": np.array([1 / res**2 * 4] * pos.size, dtype=np.float32),
            "h": h * 2,
            "res": res // 2,
            "box_x": np.float64(1),
            "box_y": np.float64(1),
        }
        ref_img = scatter(**ref_kwargs)
        repeating_img = scatter(**kwargs)
        # first quadrant image tiled should match explicitly tiled image
        diff = repeating_img - np.block([[ref_img, ref_img], [ref_img, ref_img]])
        if backend == "gpu":
            # https://github.com/SWIFTSIM/swiftsimio/issues/229
            pytest.xfail("gpu backend currently broken")
        if save:
            plt.imsave("test_image_creation.png", repeating_img.T, origin="lower")
            plt.imsave(
                "test_image_diff.png",
                diff.T,
                cmap="RdBu",
                vmin=-np.abs(diff).max(),
                vmax=np.abs(diff).max(),
            )
        assert np.allclose(
            repeating_img, np.block([[ref_img, ref_img], [ref_img, ref_img]])
        )
