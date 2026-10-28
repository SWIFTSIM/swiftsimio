"""Test the nearest-neighbour searches of the numba k-d tree."""

import numba
import numpy as np
import pytest
from swiftsimio.visualisation.kdtree import (
    k_nearest_distances,
    nearest_particles,
)

# Coordinates are shifted before the tree is built (the periodic boundary is
# moved into a gap between particles, and free axes are centred), which changes
# them by about the floating-point precision of the box size
DISTANCE_ATOL = 1e-14


def _clumpy_particles(n: int, boxsize: float, seed: int = 1) -> np.ndarray:
    """
    Make particles with densities spanning several decades, some across the edge.

    Parameters
    ----------
    n : int
        Number of particles.
    boxsize : float
        Side of the periodic box.
    seed : int
        Seed of the random numbers.

    Returns
    -------
    np.ndarray
        Positions wrapped into the box, shape ``(n, 3)``.
    """
    rng = np.random.default_rng(seed)
    n_core, n_edge = n // 3, n // 3
    # A dense core, a clump centred on a corner of the box, and a uniform field
    radius = 0.01 * boxsize * rng.random(n_core) ** 3
    direction = rng.normal(size=(n_core, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    core = 0.4 * boxsize + direction * radius[:, None]
    edge = rng.normal(0.0, 0.03 * boxsize, (n_edge, 3))
    field = rng.random((n - n_core - n_edge, 3)) * boxsize
    return np.concatenate([core, edge, field]) % boxsize


def _scipy_distances(
    positions: np.ndarray, k: int, boxsize: float | None
) -> np.ndarray:
    """
    Compute the k-th neighbour distances with scipy.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions.
    k : int
        The neighbour to find.
    boxsize : float or None
        Side of the periodic box, or ``None`` for no periodicity.

    Returns
    -------
    np.ndarray
        The distances.
    """
    spatial = pytest.importorskip("scipy.spatial")
    tree = spatial.cKDTree(positions, boxsize=boxsize)
    return tree.query(positions, k=[k])[0][:, 0]


@pytest.mark.parametrize("boxsize", [10.0, None])
@pytest.mark.parametrize("k", [1, 2, 7, 40, 150])
def test_k_nearest_distances_match_scipy(boxsize, k):
    """Check the k-d tree distances against scipy."""
    positions = _clumpy_particles(6000, 10.0)
    distances = k_nearest_distances(positions, k, boxsize)
    np.testing.assert_allclose(
        distances,
        _scipy_distances(positions, k, boxsize),
        rtol=1e-12,
        atol=DISTANCE_ATOL * 10.0,
    )


@pytest.mark.parametrize("boxsize", [10.0, None])
def test_k_nearest_distances_duplicates(boxsize):
    """Check coincident particles (k-th distance 0 for clumps of k or more)."""
    rng = np.random.default_rng(2)
    clumps = np.repeat(rng.random((50, 3)) * 10.0, rng.integers(1, 40, 50), axis=0)
    positions = np.concatenate([clumps, rng.random((2000, 3)) * 10.0])
    for k in (5, 20, 120):
        np.testing.assert_allclose(
            k_nearest_distances(positions, k, boxsize),
            _scipy_distances(positions, k, boxsize),
            rtol=1e-12,
            atol=DISTANCE_ATOL * 10.0,
        )


def test_k_nearest_distances_mixed_periodicity():
    """Check a box periodic along some axes only (a period of 0 is free)."""
    positions = _clumpy_particles(4000, 10.0, seed=3)
    distances = k_nearest_distances(positions, 32, np.array([10.0, 0.0, 10.0]))
    # scipy treats a boxsize of 0 along an axis as non-periodic too
    np.testing.assert_allclose(
        distances,
        _scipy_distances(positions, 32, np.array([10.0, 0.0, 10.0])),
        rtol=1e-12,
        atol=DISTANCE_ATOL * 10.0,
    )


def test_k_nearest_distances_small_inputs():
    """Check empty inputs, k = 1, and fewer particles than k."""
    assert k_nearest_distances(np.zeros((0, 3)), 4).shape == (0,)
    positions = np.random.default_rng(4).random((10, 3))
    np.testing.assert_array_equal(k_nearest_distances(positions, 1, 1.0), 0.0)
    assert np.all(np.isinf(k_nearest_distances(positions, 11, 1.0)))
    np.testing.assert_allclose(
        k_nearest_distances(positions, 10, 1.0),
        _scipy_distances(positions, 10, 1.0),
        rtol=1e-12,
        atol=DISTANCE_ATOL,
    )


def test_k_nearest_distances_threads():
    """Check that the distances do not depend on the number of threads."""
    positions = _clumpy_particles(20000, 10.0, seed=5)
    parallel = k_nearest_distances(positions, 32, 10.0)
    threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = k_nearest_distances(positions, 32, 10.0)
    finally:
        numba.set_num_threads(threads)
    np.testing.assert_array_equal(parallel, serial)


def _pixels(res: int, z: float, low: float = 0.0, span: float = 1.0) -> np.ndarray:
    """
    Make the pixel centres of an image, in the order of the slice backend.

    Parameters
    ----------
    res : int
        Number of pixels along each side.
    z : float
        The z coordinate of the image.
    low : float
        Lower edge of the image along x and y.
    span : float
        Side of the image.

    Returns
    -------
    np.ndarray
        The pixel centres, shape ``(res * res, 3)``.
    """
    centres = low + span * (np.arange(res) + 0.5) / res
    grids = np.meshgrid(centres, centres, [z], indexing="ij")
    return np.stack([grid.ravel() for grid in grids], axis=1)


def _distances(
    positions: np.ndarray, queries: np.ndarray, boxsize: list[float] | None
) -> np.ndarray:
    """
    Compute minimum-image distances between pairs of points.

    Parameters
    ----------
    positions : np.ndarray
        The first point of each pair.
    queries : np.ndarray
        The second point of each pair.
    boxsize : list[float] or None
        Period of each axis (0 for none), or ``None`` for no periodicity.

    Returns
    -------
    np.ndarray
        The distances.
    """
    separation = np.abs(positions - queries)
    if boxsize is not None:
        box = np.asarray(boxsize, dtype=np.float64)
        periodic = box > 0
        wrapped = separation % np.where(periodic, box, np.inf)
        separation = np.where(periodic, np.minimum(wrapped, box - wrapped), separation)
    return np.sqrt((separation**2).sum(axis=1))


@pytest.mark.parametrize("boxsize", [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], None])
@pytest.mark.parametrize(
    "image",
    [
        dict(res=128, z=0.5),
        dict(res=96, z=0.01),
        # Pixels beyond the box, which are wrapped into it
        dict(res=64, z=1.7, low=-1.0, span=3.0),
        # A zoom onto the dense core
        dict(res=64, z=0.4, low=0.35, span=0.1),
    ],
)
def test_nearest_particles_match_scipy(boxsize, image):
    """Check the nearest particle to each pixel against scipy."""
    spatial = pytest.importorskip("scipy.spatial")
    positions = _clumpy_particles(20000, 1.0)
    queries = _pixels(**image)
    nearest = nearest_particles(positions, queries, boxsize)
    expected, _ = spatial.cKDTree(positions, boxsize=boxsize).query(queries)
    # Compare distances (particles at equal distances may be chosen differently)
    np.testing.assert_allclose(
        _distances(positions[nearest], queries, boxsize),
        expected,
        rtol=1e-12,
        atol=DISTANCE_ATOL,
    )


def test_nearest_particles_small_region_of_periodic_box():
    """Check pixels far from a small region of a periodic box (nearer by the wrap)."""
    spatial = pytest.importorskip("scipy.spatial")
    positions = np.random.default_rng(2).normal(0.2, 0.01, (5000, 3))
    queries = _pixels(64, 0.9)
    nearest = nearest_particles(positions, queries, 1.0)
    expected, _ = spatial.cKDTree(positions % 1.0, boxsize=1.0).query(queries)
    np.testing.assert_allclose(
        _distances(positions[nearest], queries, [1.0] * 3),
        expected,
        rtol=1e-12,
        atol=DISTANCE_ATOL,
    )


def test_nearest_particles_threads():
    """Check that the nearest particles do not depend on the number of threads."""
    positions = _clumpy_particles(20000, 1.0, seed=3)
    queries = _pixels(128, 0.5)
    parallel = nearest_particles(positions, queries, 1.0)
    threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = nearest_particles(positions, queries, 1.0)
    finally:
        numba.set_num_threads(threads)
    np.testing.assert_array_equal(parallel, serial)


def test_nearest_particles_no_particles():
    """Check that searching no particles is an error."""
    with pytest.raises(ValueError):
        nearest_particles(np.zeros((0, 3)), _pixels(4, 0.5))
