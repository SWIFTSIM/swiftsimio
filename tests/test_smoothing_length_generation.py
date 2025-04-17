"""
Tests the smoothing length generation code.
"""

import numpy as np
import unyt as u
from swiftsimio import load, cosmo_array
from swiftsimio.visualisation.smoothing_length import generate_smoothing_lengths
from tests.helper import requires

from numpy import isclose


@requires("cosmological_volume.hdf5")
def test_generate_smoothing_length(filename):
    data = load(filename)

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

    assert isclose(
        generated_smoothing_lengths.value, smoothing_lengths.value, 0.1
    ).all()

    return


@requires("cosmological_volume.hdf5")
def test_generate_smoothing_length_faster(filename):
    data = load(filename)

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

    assert isclose(
        generated_smoothing_lengths.value, smoothing_lengths.value, 0.1
    ).all()

    return


def test_generate_smoothing_length_return_type():
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
