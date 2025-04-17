"""
Tests the smoothing length generation code.
"""

from swiftsimio import load
from swiftsimio.visualisation.smoothing_length import generate_smoothing_lengths

from numpy import isclose


def test_generate_smoothing_length(cosmological_volume):
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

    assert isclose(
        generated_smoothing_lengths.value, smoothing_lengths.value, 0.1
    ).all()

    return


def test_generate_smoothing_length_faster(cosmological_volume):
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

    assert isclose(
        generated_smoothing_lengths.value, smoothing_lengths.value, 0.1
    ).all()

    return
