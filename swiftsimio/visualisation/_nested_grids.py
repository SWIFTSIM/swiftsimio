"""
Dimension-independent helpers for the nested multi-resolution backends.

Both the 2D projection backend
(:mod:`swiftsimio.visualisation.projection_backends.nested`) and the 3D volume
render backend
(:mod:`swiftsimio.visualisation.volume_render_backends.nested_grids_scatter`)
implement the Sparse Multi-Scale Grid algorithm of Benitez-Llambay (2025,
doi:10.3847/2515-5172/addab2). This module holds the parts that do not depend
on the number of dimensions: assigning particles to hierarchy levels and
validating and converting the user inputs.
"""

import numpy as np
from numpy.typing import NDArray

from numba import njit


@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def assign_levels(
    h: NDArray[np.float32],
    m: NDArray[np.float32],
    res: int,
    ntarget: int,
    nlevels: int,
    kernel_gamma: float,
) -> tuple:
    """
    Assign a hierarchy level to every particle in a single pass.

    Replaces ``ceil(log2(support_cells / ntarget))`` with a cheap doubling
    loop, avoiding an expensive ``log2`` evaluation for every particle.  Particles
    with zero mass or negative smoothing length are marked -1 and skipped.

    Parameters
    ----------
    h : np.ndarray[float32]
        Smoothing lengths, in [0, 1] units.
    m : np.ndarray[float32]
        Particle masses or weights.
    res : int
        Finest-grid resolution along one axis.
    ntarget : int
        Target cells per kernel side at the assigned level.
    nlevels : int
        Maximum hierarchy depth.
    kernel_gamma : float
        Ratio of the kernel support radius to the smoothing length. This
        differs between the 2D and 3D kernels.

    Returns
    -------
    level_index : np.ndarray[int8]
        Per-particle level assignment; -1 for inactive particles.
    deepest : int32
        Highest level actually occupied (bounds hierarchy allocation).
    """
    level_index = np.empty(h.size, dtype=np.int8)
    support_scale = np.float32(kernel_gamma) * np.float32(res)
    target = np.float32(ntarget)
    deepest = np.int32(0)

    for particle in range(h.size):
        if m[particle] != np.float32(0.0) and h[particle] >= np.float32(0.0):
            # Double the threshold at each step rather than computing log2.
            support_cells = h[particle] * support_scale
            threshold = target
            level = np.int32(0)
            while support_cells > threshold and level < nlevels:
                level += np.int32(1)
                threshold *= np.float32(2.0)
            level_index[particle] = level
            if level > deepest:
                deepest = level
        else:
            # Skip zero-mass and unphysical particles entirely.
            level_index[particle] = np.int8(-1)

    return level_index, deepest


def validate_hierarchy(res: int, ntarget: int, nlevels: int) -> tuple:
    """
    Validate the hierarchy parameters and return them as integers.

    Parameters
    ----------
    res : int-like
        Finest-grid resolution (cells per axis).
    ntarget : int-like
        Target cells per kernel side.
    nlevels : int-like
        Maximum hierarchy depth.

    Returns
    -------
    res : int
        Validated finest-grid resolution.
    ntarget : int
        Validated target cells per kernel side.
    nlevels : int
        Validated maximum hierarchy depth.

    Raises
    ------
    ValueError
        If ``res`` is not a positive even integer divisible by ``2**nlevels``,
        if ``nlevels`` is negative, or if ``ntarget`` is not positive.
    """
    res = int(res)
    ntarget = int(ntarget)
    nlevels = int(nlevels)

    if res <= 0:
        raise ValueError(f"Pixel size must be a positive integer. Got res={res}.")
    if nlevels < 0:
        raise ValueError(
            f"The number of hierarchy levels cannot be negative. Got nlevels={nlevels}."
        )
    if res % 2 != 0:
        raise ValueError(
            f"The nested backend requires an even pixel size because each "
            f"hierarchy step coarsens the grid by exactly a factor of two. "
            f"Got res={res}. Choose an even pixel size."
        )
    required_divisor = 1 << nlevels
    if res % required_divisor != 0:
        max_levels = 0
        remaining_res = res
        while remaining_res % 2 == 0:
            max_levels += 1
            remaining_res //= 2
        lower_valid_res = (res // required_divisor) * required_divisor
        upper_valid_res = lower_valid_res + required_divisor
        alternatives = f"{upper_valid_res}"
        if lower_valid_res > 0:
            alternatives = f"{lower_valid_res} or {upper_valid_res}"
        raise ValueError(
            f"Pixel size res={res} is incompatible with nlevels={nlevels}. "
            f"A hierarchy with {nlevels} coarsening levels requires the pixel "
            f"size to be divisible by 2**{nlevels}={required_divisor}, so that "
            f"every grid has an integer size. With res={res}, use "
            f"nlevels<={max_levels}, or choose a pixel size divisible by "
            f"{required_divisor}, such as {alternatives}."
        )
    if ntarget <= 0:
        raise ValueError(f"ntarget must be greater than zero. Got ntarget={ntarget}.")

    return res, ntarget, nlevels


def prepare_particle_arrays(positions: dict, m: np.ndarray, h: np.ndarray) -> tuple:
    """
    Validate particle arrays and return typed, contiguous copies.

    Positions are promoted to float64 to avoid rounding errors in cell-index
    arithmetic at high resolution.  Masses and smoothing lengths are cast to
    float32 to match the Numba kernel signatures.

    Parameters
    ----------
    positions : dict[str, array-like]
        Particle positions keyed by axis name (e.g. ``{"x": x, "y": y}``), in
        the order they should be returned.
    m : array-like
        Particle masses or weights.
    h : array-like
        Particle smoothing lengths.

    Returns
    -------
    tuple[np.ndarray, ...]
        Contiguous float64 position arrays in the order of ``positions``,
        followed by contiguous float32 ``m`` and ``h`` arrays.

    Raises
    ------
    ValueError
        If any input is not one-dimensional, or the inputs differ in length.
    """
    names = (*positions.keys(), "m", "h")
    arrays = tuple(np.asarray(value) for value in (*positions.values(), m, h))
    joined_names = ", ".join(names[:-1]) + f", and {names[-1]}"

    invalid_shapes = [
        f"{name}.shape={array.shape}"
        for name, array in zip(names, arrays)
        if array.ndim != 1
    ]
    if invalid_shapes:
        raise ValueError(
            f"Particle inputs {joined_names} must all be one-dimensional. "
            f"Invalid inputs: {', '.join(invalid_shapes)}."
        )

    lengths = tuple(array.size for array in arrays)
    if len(set(lengths)) != 1:
        length_details = ", ".join(
            f"{name}={length}" for name, length in zip(names, lengths)
        )
        raise ValueError(
            f"Particle inputs {joined_names} must have identical lengths. "
            f"Got {length_details}."
        )

    number_of_positions = len(positions)
    return tuple(
        np.ascontiguousarray(
            array, dtype=np.float64 if i < number_of_positions else np.float32
        )
        for i, array in enumerate(arrays)
    )
