"""Test the smoothing length generation code."""

import numpy as np
import pytest
import unyt as u
from swiftsimio import load, cosmo_array
from swiftsimio.visualisation.smoothing_length import generate_smoothing_lengths

from numpy import isclose


def test_generate_smoothing_length(cosmological_volume):
    """Check that generated smoothign lengths match the ones stored with the data."""
    data = load(cosmological_volume)

    smoothing_lengths = data.gas.smoothing_lengths

    # Parameters required to generate smoothing lengths
    number_of_neighbours = int(
        round(data.metadata.hydro_scheme["Kernel target N_ngb"][0])
    )
    kernel_eta = data.metadata.hydro_scheme["Kernel eta"][0]

    kernel_gamma = ((3.0 * number_of_neighbours) / (4.0 * 3.14159)) ** (
        1 / 3
    ) / kernel_eta

    generated_smoothing_lengths = generate_smoothing_lengths(
        data.gas.coordinates,
        boxsize=data.metadata.boxsize,
        kernel_gamma=kernel_gamma,
        neighbours=number_of_neighbours,
        speedup_fac=1,
        dimension=3,
    ).to(smoothing_lengths.units)

    assert (
        isclose(generated_smoothing_lengths.value, smoothing_lengths.value, 0.2).sum()
        / generated_smoothing_lengths.size
        > 0.5
    )

    return


def test_generate_smoothing_length_faster(cosmological_volume):
    """Check that turning up the speed of the generator gives acceptable results."""
    data = load(cosmological_volume)

    smoothing_lengths = data.gas.smoothing_lengths

    # Parameters required to generate smoothing lengths
    number_of_neighbours = int(
        round(data.metadata.hydro_scheme["Kernel target N_ngb"][0])
    )
    kernel_eta = data.metadata.hydro_scheme["Kernel eta"][0]

    kernel_gamma = ((3.0 * number_of_neighbours) / (4.0 * 3.14159)) ** (
        1 / 3
    ) / kernel_eta

    generated_smoothing_lengths = generate_smoothing_lengths(
        data.gas.coordinates,
        boxsize=data.metadata.boxsize,
        kernel_gamma=kernel_gamma,
        neighbours=number_of_neighbours,
        speedup_fac=2,
        dimension=3,
    ).to(smoothing_lengths.units)

    assert (
        isclose(generated_smoothing_lengths.value, smoothing_lengths.value, 0.2).sum()
        / generated_smoothing_lengths.size
        > 0.5
    )

    return


def test_generate_smoothing_length_return_type():
    """Check that return types agree with input types (unyt vs cosmo arrays)."""
    x = cosmo_array(
        np.arange(20), u.Mpc, comoving=False, scale_factor=0.5, scale_exponent=1
    )
    xgrid, ygrid, zgrid = np.meshgrid(x, x, x)
    coords = np.vstack((xgrid.flatten(), ygrid.flatten(), zgrid.flatten())).T
    lbox = cosmo_array(
        [20, 20, 20], u.Mpc, comoving=False, scale_factor=0.5, scale_exponent=1
    )
    from_ca_input = generate_smoothing_lengths(coords, lbox, 1)
    assert from_ca_input.units == coords.units
    assert from_ca_input.comoving == coords.comoving
    assert from_ca_input.cosmo_factor == coords.cosmo_factor
    from_ua_input = generate_smoothing_lengths(u.unyt_array(coords), lbox, 1)
    assert isinstance(from_ua_input, u.unyt_array) and not isinstance(
        from_ua_input, cosmo_array
    )


@pytest.mark.parametrize("speedup_fac", [1, 2, 4])
@pytest.mark.parametrize("dimension", [2, 3])
def test_generate_smoothing_lengths_match_scipy(speedup_fac, dimension):
    """Check generate_smoothing_lengths against the scipy search it replaced."""
    spatial = pytest.importorskip("scipy.spatial")
    rng = np.random.default_rng(6)
    # A dense clump in a uniform field
    positions = (
        np.concatenate([rng.normal(10.0, 0.5, (3000, 3)), rng.random((2000, 3)) * 20.0])
        % 20.0
    )
    coordinates = cosmo_array(
        positions, u.Mpc, comoving=True, scale_factor=0.5, scale_exponent=1
    )
    boxsize = cosmo_array(
        [20.0] * 3, u.Mpc, comoving=True, scale_factor=0.5, scale_exponent=1
    )
    kernel_gamma = 1.936492
    neighbours = 48
    smoothing_lengths = generate_smoothing_lengths(
        coordinates,
        boxsize,
        kernel_gamma,
        neighbours=neighbours,
        speedup_fac=speedup_fac,
        dimension=dimension,
    )
    tree = spatial.cKDTree(positions, boxsize=20.0)
    expected = tree.query(positions, k=[neighbours // speedup_fac])[0][:, 0]
    expected *= speedup_fac ** (1 / dimension) / kernel_gamma
    assert smoothing_lengths.dtype == np.float32
    assert smoothing_lengths.units == u.Mpc
    np.testing.assert_allclose(smoothing_lengths.value, expected, rtol=1e-6)
