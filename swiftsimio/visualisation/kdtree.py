"""
Nearest-neighbour searches with a parallel k-d tree.

Two searches, both exact (the same as ``scipy.spatial.cKDTree.query``):

+ :func:`~swiftsimio.visualisation.kdtree.k_nearest_distances` finds the distance
  from every particle to its k-th nearest neighbour (for smoothing lengths);
+ :func:`~swiftsimio.visualisation.kdtree.nearest_particles` finds the particle
  nearest to each of a set of points (for nearest-neighbour images).

Notes
-----
The tree is balanced (each node splits its particles in half at the median of
its widest axis), so it is stored implicitly: node ``i`` has children
``2i + 1`` and ``2i + 2``, and every node is a contiguous range of the permuted
particles. It is built level by level, the nodes of a level in parallel, and
the coordinates are copied in tree order (one array per axis) so that the
loops over the particles of a leaf read memory in order, and the compiler can
process several particles at once.

The work is shared between a pool of Python threads running functions compiled
with ``nogil=True``, rather than numba's ``prange``: compiling a ``prange`` loop
adds seconds to the first use.

Queries go a leaf at a time, in groups of ``LEAF_GROUP`` neighbouring leaves
interleaved between threads (so that each thread gets its share of the dense
regions). Each particle is bounded by those solved before it in its leaf (or,
for the first, in the previous leaf): the k-th distance changes by at most the
distance moved, so ``d_k(q) <= d_k(a) + |q - a|``. The k-th distance is found
with a max-heap of the ``k`` nearest squared distances, searching only the nodes
nearer than that bound (the first particle of a group, with no bound, searches
from scratch).

For :func:`~swiftsimio.visualisation.kdtree.nearest_particles` the query points
go in chunks of consecutive points (as the pixels along a row of an image), and
each search starts from the particle nearest to the point before, whose distance
bounds the nearest.

Periodic boxes are handled in the metric rather than by copying particles: for
coordinates wrapped into ``[0, L)``, the separation along a periodic axis is
``min(|dx|, L - |dx|)``, and the distance from a query to a node's bounding box
also tries the box's image through the wrap. A free (non-periodic) axis has
period ``FREE_AXIS``, for which both are the non-periodic expressions. Before
building the tree, the boundary of each periodic axis is moved into the widest
gap between particles (``_periodic_cut``), and an axis the particles span less
than half of becomes free, as no distance along it can be shorter through the
wrap.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numba import config, get_num_threads, njit

LEAF_SIZE = 64
# Leaves queried one after another by a thread (a power of 2): the first
# particle of each is bounded from the previous leaf's
LEAF_GROUP = 8
# Consecutive query points searched one after another by a thread (each from
# the nearest particle to the point before it) for nearest_particles
QUERY_CHUNK = 256
# The period of an axis that is not periodic: huge but finite, so that the
# periodic separation min(|dx|, period - |dx|) is |dx| (see _separation)
FREE_AXIS = 1e300
# Floating-point optimisations for numba, except those that assume there are no
# NaNs or infinities ("nnan" and "ninf"): the searches use inf for "no bound yet"
FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}
# Fewest particles (or query points) per thread, below which fewer threads are
# used (down to one), as handing the work to a thread would take longer than it:
# for the searches, which take microseconds per point, and for shifting the
# coordinates and building the tree, which take nanoseconds per particle
SEARCH_PER_THREAD = 64
BUILD_PER_THREAD = 16384


def _tree_depth(n: int, leaf_size: int) -> int:
    """
    Find the number of levels below the root so that every leaf is small enough.

    Parameters
    ----------
    n : int
        Number of particles.
    leaf_size : int
        Largest number of particles in a leaf.

    Returns
    -------
    int
        The depth of the tree.
    """
    depth = 0
    while (n + (1 << depth) - 1) >> depth > leaf_size:
        depth += 1
    return depth


@njit(cache=True)
def _node_ranges(n: int, depth: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Compute the particle range ``[lo, hi)`` of every node of the implicit tree.

    Parameters
    ----------
    n : int
        Number of particles.
    depth : int
        Depth of the tree.

    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.int64]]
        The first particle of each node, and one past its last.
    """
    n_nodes = (1 << (depth + 1)) - 1
    lo = np.empty(n_nodes, dtype=np.int64)
    hi = np.empty(n_nodes, dtype=np.int64)
    lo[0] = 0
    hi[0] = n
    for i in range((1 << depth) - 1):
        mid = (lo[i] + hi[i]) // 2
        lo[2 * i + 1] = lo[i]
        hi[2 * i + 1] = mid
        lo[2 * i + 2] = mid
        hi[2 * i + 2] = hi[i]
    return lo, hi


@njit(cache=True)
def _bounding_box(
    pos: NDArray[np.float64],
    perm: NDArray[np.int64],
    lo: int,
    hi: int,
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    node: int,
) -> None:
    """
    Compute the bounding box of the particles of a node.

    Parameters
    ----------
    pos : NDArray[np.float64]
        Particle positions, shape ``(N, 3)``.
    perm : NDArray[np.int64]
        Particle indices in tree order.
    lo : int
        First particle (in tree order) of the node.
    hi : int
        One past the last particle of the node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes, written for ``node``.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes, written for ``node``.
    node : int
        The node.
    """
    for d in range(3):
        bmin[node, d] = np.inf
        bmax[node, d] = -np.inf
    for j in range(lo, hi):
        p = perm[j]
        for d in range(3):
            x = pos[p, d]
            if x < bmin[node, d]:
                bmin[node, d] = x
            if x > bmax[node, d]:
                bmax[node, d] = x


@njit(cache=True)
def _select(
    pos: NDArray[np.float64],
    perm: NDArray[np.int64],
    lo: int,
    hi: int,
    nth: int,
    dim: int,
) -> None:
    """
    Partition ``perm[lo:hi]`` about the particle of rank ``nth`` along ``dim``.

    Afterwards ``perm[nth]`` is the particle of that rank, with none smaller
    after it and none larger before it (quickselect, median-of-three pivots).

    Parameters
    ----------
    pos : NDArray[np.float64]
        Particle positions, shape ``(N, 3)``.
    perm : NDArray[np.int64]
        Particle indices, reordered in place.
    lo : int
        First index of the range.
    hi : int
        One past the last index of the range.
    nth : int
        The rank to select.
    dim : int
        The axis to select along.
    """
    left = lo
    right = hi - 1
    while right > left:
        mid = (left + right) // 2
        a = pos[perm[left], dim]
        b = pos[perm[mid], dim]
        c = pos[perm[right], dim]
        if (a <= b <= c) or (c <= b <= a):
            pivot = b
        elif (b <= a <= c) or (c <= a <= b):
            pivot = a
        else:
            pivot = c
        i = left
        j = right
        while i <= j:
            while pos[perm[i], dim] < pivot:
                i += 1
            while pos[perm[j], dim] > pivot:
                j -= 1
            if i <= j:
                t = perm[i]
                perm[i] = perm[j]
                perm[j] = t
                i += 1
                j -= 1
        if nth <= j:
            right = j
        elif nth >= i:
            left = i
        else:
            return


@njit(cache=True, nogil=True)
def _split_level(
    w: int,
    n_workers: int,
    pos: NDArray[np.float64],
    perm: NDArray[np.int64],
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    first: int,
    count: int,
) -> None:
    """
    Split the nodes of one level of the tree that belong to one worker.

    Each node's particles are split in half at the median of its widest axis.
    Worker ``w`` takes nodes ``first + w``, ``first + w + n_workers``, ...

    Parameters
    ----------
    w : int
        The worker.
    n_workers : int
        Number of workers.
    pos : NDArray[np.float64]
        Particle positions, shape ``(N, 3)``.
    perm : NDArray[np.int64]
        Particle indices in tree order, reordered in place.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes, written for these nodes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes, written for these nodes.
    first : int
        The first node of the level.
    count : int
        Number of nodes in the level.
    """
    for k in range(w, count, n_workers):
        node = first + k
        _bounding_box(pos, perm, lo[node], hi[node], bmin, bmax, node)
        dim = 0
        for d in range(1, 3):
            if bmax[node, d] - bmin[node, d] > bmax[node, dim] - bmin[node, dim]:
                dim = d
        mid = (lo[node] + hi[node]) // 2
        _select(pos, perm, lo[node], hi[node], mid, dim)


@njit(cache=True, nogil=True)
def _finish_leaves(
    w: int,
    n_workers: int,
    pos: NDArray[np.float64],
    perm: NDArray[np.int64],
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    first_leaf: int,
    n_leaves: int,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
) -> None:
    """
    Find the boxes of one worker's leaves, and copy their coordinates in tree order.

    Parameters
    ----------
    w : int
        The worker.
    n_workers : int
        Number of workers.
    pos : NDArray[np.float64]
        Particle positions, shape ``(N, 3)``.
    perm : NDArray[np.int64]
        Particle indices in tree order.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes, written for these leaves.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes, written for these leaves.
    first_leaf : int
        Index of the first leaf node.
    n_leaves : int
        Number of leaves.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order, written for these leaves.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order, written for these leaves.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order, written for these leaves.
    """
    for k in range(w, n_leaves, n_workers):
        node = first_leaf + k
        _bounding_box(pos, perm, lo[node], hi[node], bmin, bmax, node)
        for j in range(lo[node], hi[node]):
            xs[j] = pos[perm[j], 0]
            ys[j] = pos[perm[j], 1]
            zs[j] = pos[perm[j], 2]


def _build(
    pos: NDArray[np.float64], depth: int, pool: ThreadPoolExecutor, n_threads: int
) -> tuple:
    """
    Build the k-d tree.

    Parameters
    ----------
    pos : NDArray[np.float64]
        Particle positions, shape ``(N, 3)``.
    depth : int
        Depth of the tree.
    pool : ThreadPoolExecutor
        The threads to run on.
    n_threads : int
        Number of threads.

    Returns
    -------
    tuple
        The particle indices in tree order, the ranges ``lo`` and ``hi`` and
        the bounding boxes ``bmin`` and ``bmax`` of the nodes, and the x, y and
        z coordinates in tree order.
    """
    n = pos.shape[0]
    lo, hi = _node_ranges(n, depth)
    perm = np.arange(n)
    bmin = np.empty((lo.size, 3))
    bmax = np.empty((lo.size, 3))
    xs = np.empty(n)
    ys = np.empty(n)
    zs = np.empty(n)
    for level in range(depth):
        count = 1 << level
        _in_threads(
            pool,
            _split_level,
            min(n_threads, count),
            pos,
            perm,
            lo,
            hi,
            bmin,
            bmax,
            count - 1,
            count,
        )
    _in_threads(
        pool,
        _finish_leaves,
        n_threads,
        pos,
        perm,
        lo,
        hi,
        bmin,
        bmax,
        (1 << depth) - 1,
        1 << depth,
        xs,
        ys,
        zs,
    )
    return perm, lo, hi, bmin, bmax, xs, ys, zs


_pool = None
_pool_pid = None


def _thread_pool() -> ThreadPoolExecutor:
    """
    Get the threads the searches run on, started on first use.

    The pool is kept for later calls, as starting threads takes about as long as
    searching a few thousand particles. A process forked from this one (as by
    ``multiprocessing``) does not have its threads, so it starts its own.

    Returns
    -------
    ThreadPoolExecutor
        The pool, of ``numba.config.NUMBA_NUM_THREADS`` threads.
    """
    global _pool, _pool_pid
    if _pool is None or _pool_pid != os.getpid():
        _pool = ThreadPoolExecutor(config.NUMBA_NUM_THREADS)
        _pool_pid = os.getpid()
    return _pool


def _in_threads(
    pool: ThreadPoolExecutor, function: Callable, n_workers: int, *args: object
) -> None:
    """
    Call ``function(w, n_workers, *args)`` for every worker ``w``, in parallel.

    Parameters
    ----------
    pool : ThreadPoolExecutor
        The threads to run on.
    function : Callable
        A function compiled with ``nogil=True``, so that the workers run at once.
    n_workers : int
        Number of workers.
    *args : object
        The other arguments of ``function``.
    """
    if n_workers == 1:
        function(0, 1, *args)
        return
    # list() waits for every worker, and raises any error from one
    list(pool.map(lambda w: function(w, n_workers, *args), range(n_workers)))


def _thread_count(n: int, per_thread: int) -> int:
    """
    Choose how many threads to use for ``n`` particles or query points.

    Parameters
    ----------
    n : int
        Number of particles or query points.
    per_thread : int
        Fewest items per thread.

    Returns
    -------
    int
        At most ``numba.get_num_threads()``, with at least ``per_thread`` items
        per thread (and at least one thread).
    """
    return max(1, min(get_num_threads(), n // per_thread))


@njit(cache=True)
def _separation(dx: float, period: NDArray[np.float64], d: int) -> float:
    """
    Compute the minimum-image separation along an axis.

    For coordinates wrapped into ``[0, period[d])`` this is
    ``min(|dx|, period[d] - |dx|)``, written without an ``if`` so that the loops
    over a leaf's particles can still process several at once. On a free axis
    (period ``FREE_AXIS``) it is ``|dx|``.

    Parameters
    ----------
    dx : float
        Difference of the coordinates.
    period : NDArray[np.float64]
        Period of each axis.
    d : int
        The axis.

    Returns
    -------
    float
        The separation.
    """
    a = abs(dx)
    return min(a, period[d] - a)


@njit(cache=True)
def _axis_gap(
    low: float, high: float, x: float, period: NDArray[np.float64], d: int
) -> float:
    """
    Compute the distance from ``x`` to ``[low, high]`` along an axis.

    The distance through the periodic wrap is used if that is shorter.

    Parameters
    ----------
    low : float
        Lower end of the interval.
    high : float
        Upper end of the interval.
    x : float
        The coordinate.
    period : NDArray[np.float64]
        Period of each axis.
    d : int
        The axis.

    Returns
    -------
    float
        The distance (0 inside the interval).
    """
    if x < low:
        return min(low - x, period[d] - (high - x))
    if x > high:
        return min(x - high, period[d] - (x - low))
    return 0.0


@njit(cache=True)
def _box_min_distance2(
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    node: int,
    qx: float,
    qy: float,
    qz: float,
    period: NDArray[np.float64],
) -> float:
    """
    Compute the squared distance from a point to the nearest point of a node.

    Parameters
    ----------
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    node : int
        The node.
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    period : NDArray[np.float64]
        Period of each axis.

    Returns
    -------
    float
        The squared distance.
    """
    m = _axis_gap(bmin[node, 0], bmax[node, 0], qx, period, 0)
    d2 = m * m
    m = _axis_gap(bmin[node, 1], bmax[node, 1], qy, period, 1)
    d2 += m * m
    m = _axis_gap(bmin[node, 2], bmax[node, 2], qz, period, 2)
    d2 += m * m
    return d2


@njit(cache=True)
def _sift_down(heap: NDArray[np.float64], k: int) -> None:
    """
    Restore the max-heap after ``heap[0]`` was replaced.

    Parameters
    ----------
    heap : NDArray[np.float64]
        The heap, in its first ``k`` elements.
    k : int
        The size of the heap.
    """
    j = 0
    value = heap[0]
    while True:
        child = 2 * j + 1
        if child >= k:
            break
        if child + 1 < k and heap[child + 1] > heap[child]:
            child += 1
        if heap[child] <= value:
            break
        heap[j] = heap[child]
        j = child
    heap[j] = value


@njit(cache=True, fastmath=FASTMATH)
def _heap_query(
    qx: float,
    qy: float,
    qz: float,
    k: int,
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    first_leaf: int,
    bound2: float,
    heap: NDArray[np.float64],
    stack: NDArray[np.int64],
    stack_d2: NDArray[np.float64],
    d2: NDArray[np.float64],
) -> float:
    """
    Find the squared distance from a point to its k-th nearest particle.

    A max-heap keeps the ``k`` nearest squared distances below ``bound2``
    (``bound2`` is returned if fewer than ``k`` are): the nearer child of a node
    is searched first, and nodes whose bounding box is no nearer than the k-th
    distance so far are skipped.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    k : int
        The neighbour to find.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    first_leaf : int
        Index of the first leaf node.
    bound2 : float
        Upper bound on the squared distance.
    heap : NDArray[np.float64]
        Workspace of at least ``k`` elements.
    stack : NDArray[np.int64]
        Workspace for the nodes to visit.
    stack_d2 : NDArray[np.float64]
        Workspace for the squared distances to the nodes to visit.
    d2 : NDArray[np.float64]
        Workspace of ``LEAF_SIZE`` elements.

    Returns
    -------
    float
        The squared distance to the k-th nearest particle (or ``bound2``).
    """
    for i in range(k):
        heap[i] = bound2
    stack[0] = 0
    stack_d2[0] = 0.0
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        if stack_d2[top] >= heap[0]:
            continue
        if node >= first_leaf:
            start = lo[node]
            n = hi[node] - start
            # All the leaf's squared distances first, then their minimum: two
            # simple loops, which the compiler can run on several particles at once
            nearest = np.inf
            for j in range(n):
                dx = _separation(xs[start + j] - qx, period, 0)
                dy = _separation(ys[start + j] - qy, period, 1)
                dz = _separation(zs[start + j] - qz, period, 2)
                d2[j] = dx * dx + dy * dy + dz * dz
                nearest = min(nearest, d2[j])
            if nearest >= heap[0]:
                continue
            for j in range(n):
                if d2[j] < heap[0]:
                    heap[0] = d2[j]
                    _sift_down(heap, k)
            continue
        left = 2 * node + 1
        right = left + 1
        d_left = _box_min_distance2(bmin, bmax, left, qx, qy, qz, period)
        d_right = _box_min_distance2(bmin, bmax, right, qx, qy, qz, period)
        # Nearer child on top, so it is searched first
        if d_left <= d_right:
            near, d_near, far, d_far = left, d_left, right, d_right
        else:
            near, d_near, far, d_far = right, d_right, left, d_left
        if d_far < heap[0]:
            stack[top] = far
            stack_d2[top] = d_far
            top += 1
        if d_near < heap[0]:
            stack[top] = near
            stack_d2[top] = d_near
            top += 1
    return heap[0]


@njit(cache=True, fastmath=FASTMATH)
def _leaf_bound(
    qx: float,
    qy: float,
    qz: float,
    done_x: NDArray[np.float64],
    done_y: NDArray[np.float64],
    done_z: NDArray[np.float64],
    done_d: NDArray[np.float64],
    n_done: int,
    period: NDArray[np.float64],
) -> float:
    """
    Bound the k-th distance of a point from those of particles already solved.

    By the triangle inequality ``d_k(q) <= d_k(a) + |q - a|`` for every solved
    particle ``a``.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    done_x : NDArray[np.float64]
        The x coordinates of the solved particles.
    done_y : NDArray[np.float64]
        The y coordinates of the solved particles.
    done_z : NDArray[np.float64]
        The z coordinates of the solved particles.
    done_d : NDArray[np.float64]
        The k-th distances of the solved particles.
    n_done : int
        Number of solved particles.
    period : NDArray[np.float64]
        Period of each axis.

    Returns
    -------
    float
        The upper bound (``inf`` if none is solved).
    """
    bound = np.inf
    for a in range(n_done):
        dx = _separation(done_x[a] - qx, period, 0)
        dy = _separation(done_y[a] - qy, period, 1)
        dz = _separation(done_z[a] - qz, period, 2)
        bound = min(bound, done_d[a] + np.sqrt(dx * dx + dy * dy + dz * dz))
    return bound


@njit(cache=True, nogil=True)
def _query_worker(
    w: int,
    n_workers: int,
    group: int,
    k: int,
    perm: NDArray[np.int64],
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    depth: int,
    out: NDArray[np.float64],
) -> None:
    """
    Find the k-th distances of the particles of one worker.

    Worker ``w`` takes the leaves in groups of ``group`` neighbouring leaves (a
    subtree): groups ``w``, ``w + n_workers``, ...

    Parameters
    ----------
    w : int
        The worker.
    n_workers : int
        Number of workers.
    group : int
        Number of neighbouring leaves queried one after another.
    k : int
        The neighbour to find (``2 <= k <= N``).
    perm : NDArray[np.int64]
        Particle indices in tree order.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    depth : int
        Depth of the tree.
    out : NDArray[np.float64]
        The k-th distances, written for this worker's particles.
    """
    first_leaf = (1 << depth) - 1
    n_leaves = 1 << depth
    args = (lo, hi, bmin, bmax, period, xs, ys, zs, first_leaf)
    heap = np.empty(k, dtype=xs.dtype)
    stack = np.empty(2 * depth + 4, dtype=np.int64)
    stack_d2 = np.empty(2 * depth + 4, dtype=xs.dtype)
    d2_leaf = np.empty(LEAF_SIZE, dtype=xs.dtype)
    done_x = np.empty(LEAF_SIZE, dtype=xs.dtype)
    done_y = np.empty(LEAF_SIZE, dtype=xs.dtype)
    done_z = np.empty(LEAF_SIZE, dtype=xs.dtype)
    done_d = np.empty(LEAF_SIZE, dtype=xs.dtype)
    n_done = 0
    for leaf in range(first_leaf, first_leaf + n_leaves):
        if (leaf - first_leaf) // group % n_workers != w:
            continue
        if (leaf - first_leaf) % group == 0:
            # A new group: nothing solved nearby yet
            n_done = 0
        first = True
        for j in range(lo[leaf], hi[leaf]):
            qx = xs[j]
            qy = ys[j]
            qz = zs[j]
            # Bound from the leaf's particles solved so far, or for its first
            # from the previous leaf's (none for the first leaf of a group)
            bound = _leaf_bound(
                qx, qy, qz, done_x, done_y, done_z, done_d, n_done, period
            )
            if first:
                first = False
                n_done = 0
            if bound == 0.0:
                # A solved particle at the same place, with k there: so has q
                d2 = 0.0
            else:
                # A margin for the rounding of the distances
                bound2 = (bound * (1.0 + 1e-9)) ** 2
                d2 = _heap_query(
                    qx, qy, qz, k, *args, bound2, heap, stack, stack_d2, d2_leaf
                )
                if d2 >= bound2:
                    # Fewer than k inside the bound (only from rounding)
                    d2 = _heap_query(
                        qx, qy, qz, k, *args, np.inf, heap, stack, stack_d2, d2_leaf
                    )
            d = np.sqrt(d2)
            out[perm[j]] = d
            done_x[n_done] = qx
            done_y[n_done] = qy
            done_z[n_done] = qz
            done_d[n_done] = d
            n_done += 1


@njit(cache=True, fastmath=FASTMATH)
def _nearest_query(
    qx: float,
    qy: float,
    qz: float,
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    first_leaf: int,
    best2: float,
    best: int,
    stack: NDArray[np.int64],
    stack_d2: NDArray[np.float64],
    d2: NDArray[np.float64],
) -> int:
    """
    Find the particle nearest to a point.

    The nearer child of a node is searched first, and nodes no nearer than the
    nearest particle so far are skipped. The search starts from a particle
    already known (the nearest to the previous point), whose distance bounds
    that of the nearest.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    first_leaf : int
        Index of the first leaf node.
    best2 : float
        Squared distance to the particle ``best`` (``inf`` if none).
    best : int
        A particle (in tree order) to start from (-1 if none).
    stack : NDArray[np.int64]
        Workspace for the nodes to visit.
    stack_d2 : NDArray[np.float64]
        Workspace for the squared distances to the nodes to visit.
    d2 : NDArray[np.float64]
        Workspace of ``LEAF_SIZE`` elements.

    Returns
    -------
    int
        The nearest particle, in tree order.
    """
    stack[0] = 0
    stack_d2[0] = 0.0
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        if stack_d2[top] >= best2:
            continue
        if node >= first_leaf:
            start = lo[node]
            n = hi[node] - start
            # All the leaf's squared distances first, then their minimum: two
            # simple loops, which the compiler can run on several particles at once
            nearest = np.inf
            for j in range(n):
                dx = _separation(xs[start + j] - qx, period, 0)
                dy = _separation(ys[start + j] - qy, period, 1)
                dz = _separation(zs[start + j] - qz, period, 2)
                d2[j] = dx * dx + dy * dy + dz * dz
                nearest = min(nearest, d2[j])
            if nearest < best2:
                for j in range(n):
                    if d2[j] == nearest:
                        best = start + j
                        break
                best2 = nearest
            continue
        left = 2 * node + 1
        right = left + 1
        d_left = _box_min_distance2(bmin, bmax, left, qx, qy, qz, period)
        d_right = _box_min_distance2(bmin, bmax, right, qx, qy, qz, period)
        # Nearer child on top, so it is searched first
        if d_left <= d_right:
            near, d_near, far, d_far = left, d_left, right, d_right
        else:
            near, d_near, far, d_far = right, d_right, left, d_left
        if d_far < best2:
            stack[top] = far
            stack_d2[top] = d_far
            top += 1
        if d_near < best2:
            stack[top] = near
            stack_d2[top] = d_near
            top += 1
    return best


@njit(cache=True, nogil=True)
def _nearest_worker(
    w: int,
    n_workers: int,
    query_x: NDArray[np.float64],
    query_y: NDArray[np.float64],
    query_z: NDArray[np.float64],
    perm: NDArray[np.int64],
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    depth: int,
    out: NDArray[np.int64],
) -> None:
    """
    Find the particle nearest to each of one worker's query points.

    The points go in chunks of ``QUERY_CHUNK`` consecutive points interleaved
    between the workers; each point starts its search from the nearest particle
    to the one before it in its chunk.

    Parameters
    ----------
    w : int
        The worker.
    n_workers : int
        Number of workers.
    query_x : NDArray[np.float64]
        The query points' x coordinates.
    query_y : NDArray[np.float64]
        The query points' y coordinates.
    query_z : NDArray[np.float64]
        The query points' z coordinates.
    perm : NDArray[np.int64]
        Particle indices in tree order.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    depth : int
        Depth of the tree.

    out : NDArray[np.int64]
        The index of the nearest particle to each query point, written for this
        worker's points.
    """
    m = query_x.size
    first_leaf = (1 << depth) - 1
    n_chunks = (m + QUERY_CHUNK - 1) // QUERY_CHUNK
    stack = np.empty(2 * depth + 4, dtype=np.int64)
    stack_d2 = np.empty(2 * depth + 4, dtype=xs.dtype)
    d2 = np.empty(LEAF_SIZE, dtype=xs.dtype)
    for chunk in range(w, n_chunks, n_workers):
        best = -1
        for i in range(chunk * QUERY_CHUNK, min(m, (chunk + 1) * QUERY_CHUNK)):
            qx = query_x[i]
            qy = query_y[i]
            qz = query_z[i]
            best2 = np.inf
            if best >= 0:
                dx = _separation(xs[best] - qx, period, 0)
                dy = _separation(ys[best] - qy, period, 1)
                dz = _separation(zs[best] - qz, period, 2)
                best2 = dx * dx + dy * dy + dz * dz
            best = _nearest_query(
                qx,
                qy,
                qz,
                lo,
                hi,
                bmin,
                bmax,
                period,
                xs,
                ys,
                zs,
                first_leaf,
                best2,
                best,
                stack,
                stack_d2,
                d2,
            )
            out[i] = perm[best]


def _periodic_cut(x: NDArray[np.float64], period: float, bins: int = 4096) -> float:
    """
    Choose where to put the boundary of a periodic axis.

    Periodic distances do not depend on where the boundary is, but the tree
    does: a clump of particles across the boundary would be cut into pieces at
    opposite faces, and nodes holding both would have bounding boxes as wide as
    the box, which prune nothing. The boundary is put in the middle of the
    widest run of empty bins of a histogram of ``x`` (0 if none is empty).

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinates along the axis.
    period : float
        Period of the axis.
    bins : int, optional
        Number of bins of the histogram. Default is ``4096``.

    Returns
    -------
    float
        The coordinate of the boundary.
    """
    which = np.minimum((x % period) * (bins / period), bins - 1).astype(np.int64)
    empty = np.bincount(which, minlength=bins) == 0
    if not empty.any():
        return 0.0
    # Runs of empty bins, from a filled bin, so none wraps around the end
    start = int(np.argmin(empty))
    rolled = np.roll(empty, -start)
    edges = np.flatnonzero(np.diff(np.r_[0, rolled.astype(np.int8), 0]))
    run_starts, run_ends = edges[::2], edges[1::2]
    widest = int(np.argmax(run_ends - run_starts))
    middle = 0.5 * (run_starts[widest] + run_ends[widest]) + start
    return (middle % bins) * (period / bins)


@njit(cache=True, nogil=True)
def _shift_axis(
    w: int,
    n_workers: int,
    x: NDArray[np.float64],
    offset: float,
    period: float,
    out: NDArray[np.float64],
) -> None:
    """
    Shift one worker's block of coordinates, and wrap them into ``[0, period)``.

    Parameters
    ----------
    w : int
        The worker.
    n_workers : int
        Number of workers.
    x : NDArray[np.float64]
        The coordinates.
    offset : float
        Subtracted from the coordinates.
    period : float
        Period of the axis (0 for no wrapping).
    out : NDArray[np.float64]
        The shifted coordinates, written for this worker's block.
    """
    block = (x.size + n_workers - 1) // n_workers
    for i in range(w * block, min(x.size, (w + 1) * block)):
        v = x[i] - offset
        if period > 0.0:
            v %= period
            # v % L rounds up to L for tiny negative v
            if v >= period:
                v = 0.0
        out[i] = v


def _shift_into_frame(
    positions: NDArray[np.float64],
    boxsize: float | NDArray[np.floating] | None,
    pool: ThreadPoolExecutor,
    n_threads: int,
    queries: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """
    Shift the particles (and query points) into the frame the tree is built in.

    On periodic axes the coordinates are wrapped into ``[0, period)``, which the
    periodic separation ``min(|dx|, period - |dx|)`` needs, with the boundary moved into the widest gap
    between particles (distances do not change); on free axes they are centred,
    for the precision of the distances. A periodic axis along which all the
    points are within half a period of each other becomes free, as no distance
    along it is shorter through the wrap (the case for a small region of a large
    box).

    Parameters
    ----------
    positions : NDArray[np.float64]
        Particle positions, shape ``(N, 3)`` with ``N > 0``.
    boxsize : float or NDArray[np.floating] or None
        Period of each axis (``None``, or a period ``<= 0``, for a free axis).
    pool : ThreadPoolExecutor
        The threads to run on.
    n_threads : int
        Number of threads.
    queries : NDArray[np.float64], optional
        Query points, shape ``(M, 3)``, shifted with the particles. Default is
        ``None``.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]
        The shifted positions, shape ``(N, 3)``; the period of each axis
        (``FREE_AXIS`` if free); and the shifted query points, one row per axis,
        shape ``(3, M)`` (``None`` if none were given).
    """
    period = np.full(3, FREE_AXIS)
    if boxsize is not None:
        box = np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,))
        period[box > 0] = box[box > 0]
    shifted = np.empty_like(positions)
    shifted_queries = None if queries is None else np.empty((3, queries.shape[0]))
    for d in range(3):
        x = np.ascontiguousarray(positions[:, d])
        column = np.empty_like(x)
        if period[d] < FREE_AXIS:
            offset = _periodic_cut(x, period[d])
            _in_threads(pool, _shift_axis, n_threads, x, offset, period[d], column)
            low, high = column.min(), column.max()
        else:
            offset = 0.5 * (x.min() + x.max())
            _in_threads(pool, _shift_axis, n_threads, x, offset, 0.0, column)
        shifted[:, d] = column
        if queries is not None:
            q = np.ascontiguousarray(queries[:, d])
            wrap = period[d] if period[d] < FREE_AXIS else 0.0
            _in_threads(
                pool, _shift_axis, n_threads, q, offset, wrap, shifted_queries[d]
            )
            if period[d] < FREE_AXIS:
                # A query outside the particles' range can be nearer through
                # the wrap, so all the points must be within half a period
                low = min(low, shifted_queries[d].min())
                high = max(high, shifted_queries[d].max())
        if period[d] < FREE_AXIS and high - low <= 0.5 * period[d]:
            period[d] = FREE_AXIS
    return shifted, period, shifted_queries


def k_nearest_distances(
    positions: NDArray[np.floating],
    k: int,
    boxsize: float | NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """
    Compute the distance from every particle to its k-th nearest neighbour.

    The particle itself is included (so ``k=1`` gives 0), as in
    ``scipy.spatial.cKDTree(positions, boxsize=boxsize).query(positions, k=[k])``.
    Runs on up to ``numba.get_num_threads()`` threads (fewer for small inputs).

    Parameters
    ----------
    positions : NDArray[np.floating]
        Particle positions, shape ``(N, 3)``.
    k : int
        The neighbour to find.
    boxsize : float or NDArray[np.floating] or None, optional
        Periodic box: distances are minimum-image distances along every axis
        with a positive period (the coordinates are wrapped into it). ``None``,
        or a period ``<= 0``, leaves that axis non-periodic. Default is ``None``.

    Returns
    -------
    NDArray[np.float64]
        The distances, shape ``(N,)`` (``inf`` where there are fewer than ``k``
        particles).
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    if n == 0:
        return np.zeros(0)
    out = np.zeros(n)
    if k > n:
        out[:] = np.inf
        return out
    if k <= 1:
        return out
    pool = _thread_pool()
    n_threads = _thread_count(n, BUILD_PER_THREAD)
    shifted, period, _ = _shift_into_frame(positions, boxsize, pool, n_threads)
    depth = _tree_depth(n, LEAF_SIZE)
    perm, lo, hi, bmin, bmax, xs, ys, zs = _build(shifted, depth, pool, n_threads)
    # A leaf at a time, in groups of LEAF_GROUP neighbouring leaves (a
    # subtree), the groups interleaved between the workers (so that the work
    # on the densest region is shared between them)
    _in_threads(
        pool,
        _query_worker,
        _thread_count(n, SEARCH_PER_THREAD),
        min(LEAF_GROUP, 1 << depth),
        int(k),
        perm,
        lo,
        hi,
        bmin,
        bmax,
        period,
        xs,
        ys,
        zs,
        depth,
        out,
    )
    return out


def nearest_particles(
    positions: NDArray[np.floating],
    queries: NDArray[np.floating],
    boxsize: float | NDArray[np.floating] | None = None,
) -> NDArray[np.int64]:
    """
    Find the particle nearest to each of a set of points.

    The same as ``scipy.spatial.cKDTree(positions, boxsize=boxsize).query(queries)``
    (up to the choice between particles at equal distances). Consecutive query
    points should be near each other (as the pixels of an image), as each search
    starts from the particle nearest to the point before. Runs on up to
    ``numba.get_num_threads()`` threads (fewer for small inputs).

    Parameters
    ----------
    positions : NDArray[np.floating]
        Particle positions, shape ``(N, 3)`` with ``N > 0``.
    queries : NDArray[np.floating]
        Query points, shape ``(M, 3)``.
    boxsize : float or NDArray[np.floating] or None, optional
        Periodic box: distances are minimum-image distances along every axis
        with a positive period (the coordinates are wrapped into it). ``None``,
        or a period ``<= 0``, leaves that axis non-periodic. Default is ``None``.

    Returns
    -------
    NDArray[np.int64]
        The index of the nearest particle to each query point, shape ``(M,)``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    if positions.shape[0] == 0:
        raise ValueError("Cannot find the nearest of no particles.")
    out = np.empty(queries.shape[0], dtype=np.int64)
    n, m = positions.shape[0], queries.shape[0]
    pool = _thread_pool()
    shifted, period, shifted_queries = _shift_into_frame(
        positions, boxsize, pool, _thread_count(max(n, m), BUILD_PER_THREAD), queries
    )
    depth = _tree_depth(n, LEAF_SIZE)
    tree = _build(shifted, depth, pool, _thread_count(n, BUILD_PER_THREAD))
    _in_threads(
        pool,
        _nearest_worker,
        _thread_count(m, SEARCH_PER_THREAD),
        shifted_queries[0],
        shifted_queries[1],
        shifted_queries[2],
        *tree[:5],
        period,
        *tree[5:],
        depth,
        out,
    )
    return out
