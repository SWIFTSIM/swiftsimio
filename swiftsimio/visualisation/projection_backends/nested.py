"""
Nested multi-resolution projection backend for SPH data.

Particles whose smoothing kernels span many pixels at the target resolution
are scattered onto successively coarser sub-grids and then bilinearly
upsampled back to the target resolution.  This bounds the per-particle work
regardless of smoothing length, giving large speedups for simulations with
wide kernel-size distributions (e.g. zoom-in runs or IGM particles).

This is the 2D counterpart of
:mod:`swiftsimio.visualisation.volume_render_backends.nested_grids_scatter`.

Notes
-----
Algorithm:

1. For each particle compute how many finest-grid cells its compact-support
   kernel spans::

       support_cells = kernel_gamma * h * res

2. Assign particle to level ``L`` where::

       L = clip(ceil(log2(support_cells / ntarget)), 0, nlevels)

   Level 0 is the finest grid (``res``); level ``L`` uses a grid of size
   ``res // 2**L``.  At their assigned level every kernel spans roughly
   ``ntarget`` cells, bounding the number of pixels visited per particle.

3. Scatter each level's particles onto their coarse grid.

4. Collapse the hierarchy coarse-to-fine: bilinearly upsample each coarse
   grid and accumulate it into the next finer grid.

5. Return the finest grid.

The algorithm follows a novel Sparse Multi-Scale Grid algorithm
(as described in Benitez-Llambay 2025, doi:10.3847/2515-5172/addab2)
to place particles on a grid using an Adaptive Mesh Refinement (AMR)
approach.

Resolution constraints: ``res`` must be even and divisible by ``2**nlevels``
so that each level has an integer grid size.  With the default ``nlevels=4``
this requires ``res`` to be a multiple of 16.
"""

from math import ceil, floor, sqrt

import numpy as np
from numpy.typing import NDArray

from numba import get_num_threads, njit, prange

from swiftsimio.visualisation._nested_grids import (
    assign_levels,
    prepare_particle_arrays,
    validate_hierarchy,
)
from swiftsimio.visualisation.projection_backends.kernels import (
    kernel_gamma,
    kernel_constant,
)


@njit(
    fastmath=True,
    cache=True,
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    inline="always",
)
def _deposit_particle_flat(
    destination: NDArray[np.float32],
    level_offset: int,
    level_res: int,
    x_pos: np.float64,
    y_pos: np.float64,
    mass_ica: np.float32,
    weighted_prefactor: np.float32,
    radius_cells_64: np.float64,
    level: int,
    bounds_min: NDArray[np.int32],
    bounds_max: NDArray[np.int32],
) -> None:
    """
    Deposit one particle onto the flat destination array at the given level.

    The destination is a flat (1-D) view shared across all levels; ``level_offset``
    locates the start of this level's sub-array.  Bounds arrays track the
    bounding box of occupied coarse cells so that the collapse step only
    visits cells that were actually written.

    All particle-invariant quantities (``weighted_prefactor``, ``radius_cells_64``,
    ``mass_ica``) are pre-computed by the caller before the periodic-image loop.
    Kernel distances are evaluated in cell-space coordinates so that the inner-loop
    increment is exactly ``1.0`` and the y-loop runs without a branch.

    Parameters
    ----------
    destination : np.ndarray[float32]
        Flat output array (finest or coarse, depending on level).
    level_offset : int64
        Index of the first cell belonging to this level in ``destination``.
    level_res : int32
        Number of cells along one axis at this level.
    x_pos : float64
        Particle x coordinate in [0, 1].
    y_pos : float64
        Particle y coordinate in [0, 1].
    mass_ica : float32
        ``mass * level_res**2`` — used by the sub-pixel deposit path.
    weighted_prefactor : float32
        ``mass * (7 / np.pi) * H**-2``, where ``H = kernel_gamma * hsml`` is
        the kernel support radius — kernel amplitude folded into one scalar.
    radius_cells_64 : float64
        ``kernel_gamma * hsml * level_res`` — kernel radius in cell units, kept
        in float64 for accurate bounding-box derivation.
    level : int32
        Hierarchy level of this particle.
    bounds_min : np.ndarray[int32, shape=(nlevels+1, 2)]
        Per-level bounding-box minimum cell indices (updated in-place).
    bounds_max : np.ndarray[int32, shape=(nlevels+1, 2)]
        Per-level bounding-box maximum cell indices (updated in-place).
    """
    # Positions scaled to cell-index space; float64 for accurate index derivation.
    scaled_x_64 = np.float64(level_res) * x_pos
    scaled_y_64 = np.float64(level_res) * y_pos

    radius_cells_f32 = np.float32(radius_cells_64)
    maximal_index = level_res - 1

    # Sub-pixel fast path: kernel spans less than one cell half-width.
    # Deposit the full particle weight into the containing cell to preserve mass.
    if radius_cells_f32 < np.float32(0.5):
        px = np.int32(floor(scaled_x_64))
        py = np.int32(floor(scaled_y_64))
        if 0 <= px <= maximal_index and 0 <= py <= maximal_index:
            destination[level_offset + px * level_res + py] += mass_ica
            if level != 0:
                if px < bounds_min[level, 0]:
                    bounds_min[level, 0] = px
                if py < bounds_min[level, 1]:
                    bounds_min[level, 1] = py
                if px > bounds_max[level, 0]:
                    bounds_max[level, 0] = px
                if py > bounds_max[level, 1]:
                    bounds_max[level, 1] = py
        return

    # Exact kernel bounding box in cell space.  Cell i has its centre at
    # i + 0.5; we include cells where that centre falls strictly inside the
    # kernel compact support.
    x_start = np.int32(floor(scaled_x_64 - radius_cells_64 - 0.5)) + 1
    x_stop = np.int32(ceil(scaled_x_64 + radius_cells_64 - 0.5))
    if x_start < 0:
        x_start = 0
    if x_stop > level_res:
        x_stop = level_res

    y_start = np.int32(floor(scaled_y_64 - radius_cells_64 - 0.5)) + 1
    y_stop = np.int32(ceil(scaled_y_64 + radius_cells_64 - 0.5))
    if y_start < 0:
        y_start = 0
    if y_stop > level_res:
        y_stop = level_res

    if x_start >= x_stop or y_start >= y_stop:
        return

    if level != 0:
        if x_start < bounds_min[level, 0]:
            bounds_min[level, 0] = x_start
        if y_start < bounds_min[level, 1]:
            bounds_min[level, 1] = y_start
        if x_stop - 1 > bounds_max[level, 0]:
            bounds_max[level, 0] = x_stop - 1
        if y_stop - 1 > bounds_max[level, 1]:
            bounds_max[level, 1] = y_stop - 1

    # float32 particle positions for kernel arithmetic; inner-loop distances
    # accumulate by exactly 1.0 rather than pixel_width, removing that multiply.
    scaled_y_f32 = np.float32(scaled_y_64)
    radius_cells_2_64 = radius_cells_64 * radius_cells_64
    inverse_radius_cells = np.float32(1.0) / radius_cells_f32

    # Single-cell fast path: evaluate the kernel once, no loop overhead.
    if x_stop - x_start == 1 and y_stop - y_start == 1:
        dx = np.float32(np.float64(x_start) + 0.5 - scaled_x_64)
        dy = np.float32(y_start) + np.float32(0.5) - scaled_y_f32
        ratio = sqrt(dx * dx + dy * dy) * inverse_radius_cells
        one_minus_ratio = np.float32(1.0) - ratio
        one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
        destination[level_offset + x_start * level_res + y_start] += (
            weighted_prefactor
            * one_minus_ratio_2
            * one_minus_ratio_2
            * (np.float32(1.0) + np.float32(4.0) * ratio)
        )
        return

    # General kernel loop in cell-space coordinates.
    # Float64 is used only for per-strip analytic bound derivation; the kernel
    # arithmetic stays in float32.  The innermost y-loop is branch-free because
    # tight_y bounds already exclude cells outside the kernel support.
    # Wendland-C2 (2D): W(r,H) = (7/π) H⁻² (1 - r/H)⁴ (1 + 4r/H)
    dx_64 = np.float64(x_start) + 0.5 - scaled_x_64
    dx = np.float32(dx_64)
    for cell_x in range(x_start, x_stop):
        dx_2 = dx * dx

        # Tight y-strip: analytic bound given the current x-distance.
        rem_x_sq = radius_cells_2_64 - dx_64 * dx_64
        if rem_x_sq < 0.0:
            rem_x_sq = 0.0
        rem_x_64 = sqrt(rem_x_sq)
        tight_y_start = np.int32(floor(scaled_y_64 - rem_x_64 - 0.5)) + 1
        tight_y_stop = np.int32(ceil(scaled_y_64 + rem_x_64 - 0.5))
        if tight_y_start < y_start:
            tight_y_start = y_start
        if tight_y_stop > y_stop:
            tight_y_stop = y_stop

        if tight_y_start < tight_y_stop:
            flat_cell = level_offset + cell_x * level_res + tight_y_start
            dy = np.float32(tight_y_start) + np.float32(0.5) - scaled_y_f32

            for cell_y in range(tight_y_start, tight_y_stop):
                ratio = sqrt(dx_2 + dy * dy) * inverse_radius_cells
                one_minus_ratio = np.float32(1.0) - ratio
                one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
                destination[flat_cell] += (
                    weighted_prefactor
                    * one_minus_ratio_2
                    * one_minus_ratio_2
                    * (np.float32(1.0) + np.float32(4.0) * ratio)
                )
                flat_cell += 1
                dy += np.float32(1.0)
        dx_64 += 1.0
        dx += np.float32(1.0)


@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _scatter_particles(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    m: NDArray[np.float32],
    h: NDArray[np.float32],
    level_index: NDArray[np.int8],
    level_offsets: NDArray[np.int64],
    level_resolutions: NDArray[np.int32],
    finest: NDArray[np.float32],
    coarse: NDArray[np.float32],
    box_x: np.float64,
    box_y: np.float64,
    bounds_min: NDArray[np.int32],
    bounds_max: NDArray[np.int32],
) -> None:
    """
    Scatter all active particles onto their assigned hierarchy level.

    Loops over particles and, for each, over the periodic images of the particle
    whose kernel support overlaps [0, 1].  When a box dimension is 0.0 the
    shift range collapses to a single image, so both periodic and
    non-periodic cases are handled by the same loop.  Per-particle image
    ranges are computed analytically so interior particles (whose kernels do
    not reach any periodic boundary) make only one call to
    ``_deposit_particle_flat``, saving the overhead of 8 no-op calls.

    Parameters
    ----------
    x : np.ndarray[float64]
        Particle x-positions in [0, 1].
    y : np.ndarray[float64]
        Particle y-positions in [0, 1].
    m : np.ndarray[float32]
        Particle masses or weights.
    h : np.ndarray[float32]
        Particle smoothing lengths.
    level_index : np.ndarray[int8]
        Pre-assigned hierarchy level per particle; -1 = skip.
    level_offsets : np.ndarray[int64]
        Flat-array start index of each level.
    level_resolutions : np.ndarray[int32]
        Cell count per axis at each level.
    finest : np.ndarray[float32]
        Flat level-0 accumulator of size ``res**2``.
    coarse : np.ndarray[float32]
        Flat accumulator for levels 1+ (total coarse cells).
    box_x : float64
        Box size in x; 0.0 means no wrapping in x.
    box_y : float64
        Box size in y; 0.0 means no wrapping in y.
    bounds_min : np.ndarray[int32, shape=(nlevels+1, 2)]
        Per-level bounding-box minimum (updated in-place).
    bounds_max : np.ndarray[int32, shape=(nlevels+1, 2)]
        Per-level bounding-box maximum (updated in-place).
    """
    finest_cells = finest.size

    # Global shift bounds: the widest range any particle could need.
    # Per-particle ranges are derived analytically below and clipped to these.
    xshift_min = np.int32(0) if box_x == 0.0 else np.int32(-1)
    yshift_min = np.int32(0) if box_y == 0.0 else np.int32(-1)
    xshift_max = np.int32(1) if box_x == 0.0 else np.int32(ceil(1.0 / box_x) + 1)
    yshift_max = np.int32(1) if box_y == 0.0 else np.int32(ceil(1.0 / box_y) + 1)

    for particle in range(h.size):
        level = np.int32(level_index[particle])
        if level < 0:
            continue

        mass = m[particle]
        hsml = h[particle]
        if level == 0:
            destination = finest
            level_offset = np.int64(0)
        else:
            destination = coarse
            level_offset = level_offsets[level] - finest_cells

        level_res = level_resolutions[level]
        original_x = x[particle]
        original_y = y[particle]

        # Compute all per-particle constants once, outside the periodic image loop.
        kernel_width = np.float32(kernel_gamma) * hsml
        inverse_kernel_width = np.float32(1.0) / kernel_width
        weighted_prefactor = (
            mass
            * np.float32(kernel_constant)
            * inverse_kernel_width
            * inverse_kernel_width
        )
        float_res = np.float32(level_res)
        mass_ica = mass * float_res * float_res
        kernel_width_64 = np.float64(kernel_width)
        radius_cells_64 = kernel_width_64 * np.float64(level_res)

        # Per-particle image range: only shifts whose kernel overlaps [0, 1].
        # For an interior particle this collapses to a single image (shift = 0).
        if box_x != 0.0:
            local_x_min = np.int32(ceil((-kernel_width_64 - original_x) / box_x))
            local_x_max = (
                np.int32(floor((1.0 + kernel_width_64 - original_x) / box_x)) + 1
            )
            if local_x_min < xshift_min:
                local_x_min = xshift_min
            if local_x_max > xshift_max:
                local_x_max = xshift_max
        else:
            local_x_min = np.int32(0)
            local_x_max = np.int32(1)

        if box_y != 0.0:
            local_y_min = np.int32(ceil((-kernel_width_64 - original_y) / box_y))
            local_y_max = (
                np.int32(floor((1.0 + kernel_width_64 - original_y) / box_y)) + 1
            )
            if local_y_min < yshift_min:
                local_y_min = yshift_min
            if local_y_max > yshift_max:
                local_y_max = yshift_max
        else:
            local_y_min = np.int32(0)
            local_y_max = np.int32(1)

        for xshift in range(local_x_min, local_x_max):
            x_pos = original_x + xshift * box_x
            for yshift in range(local_y_min, local_y_max):
                _deposit_particle_flat(
                    destination,
                    level_offset,
                    level_res,
                    x_pos,
                    original_y + yshift * box_y,
                    mass_ica,
                    weighted_prefactor,
                    radius_cells_64,
                    level,
                    bounds_min,
                    bounds_max,
                )


@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _collapse_serial_flat(
    finest: NDArray[np.float32],
    coarse: NDArray[np.float32],
    level_offsets: NDArray[np.int64],
    level_resolutions: NDArray[np.int32],
    nlevels: np.int32,
    bounds_min: NDArray[np.int32],
    bounds_max: NDArray[np.int32],
) -> None:
    """
    Collapse the hierarchy coarse-to-fine by bilinear upsampling.

    Starting from the deepest occupied level, each coarse grid is bilinearly
    upsampled by a factor of two and accumulated into the next finer grid,
    until everything has been folded into the finest grid.  Only the bounding
    box of occupied coarse cells (plus a one-cell halo) is visited, so the cost
    is approximately proportional to the number of particles rather than the
    grid area.

    The bilinear weights follow the stencil used by the 3D nested backend:
    an even fine cell at index ``2i`` draws 3/4 from coarse cell ``i`` and
    1/4 from coarse cell ``i-1``; an odd fine cell at ``2i+1`` draws 3/4
    from ``i`` and 1/4 from ``i+1``.

    Parameters
    ----------
    finest : np.ndarray[float32]
        Flat level-0 accumulator (modified in-place).
    coarse : np.ndarray[float32]
        Flat accumulator for levels 1+ (read and modified in-place).
    level_offsets : np.ndarray[int64]
        Start index of each level in the flat arrays.
    level_resolutions : np.ndarray[int32]
        Cell count per axis at each level.
    nlevels : int
        Deepest occupied level (collapse stops here).
    bounds_min : np.ndarray[int32, shape=(nlevels+1, 2)]
        Per-level bounding-box minimum populated during scatter.
    bounds_max : np.ndarray[int32, shape=(nlevels+1, 2)]
        Per-level bounding-box maximum populated during scatter.
    """
    finest_cells = finest.size

    for level in range(nlevels, 0, -1):
        # Skip levels that received no particles (bounds_max stays at -1).
        if bounds_max[level, 0] < 0:
            continue

        coarse_res = level_resolutions[level]
        fine_res = level_resolutions[level - 1]
        coarse_max = coarse_res - 1
        source_offset = level_offsets[level] - finest_cells

        if level == 1:
            destination = finest
            destination_offset = 0
        else:
            destination = coarse
            destination_offset = level_offsets[level - 1] - finest_cells

        # One occupied coarse cell can influence fine indices 2*i-1 through
        # 2*i+2. Restricting to this box is exact because all cells outside
        # the tracked source box are known to be zero.
        fine_x_start = (bounds_min[level, 0] << 1) - 1
        fine_y_start = (bounds_min[level, 1] << 1) - 1
        fine_x_stop = (bounds_max[level, 0] << 1) + 3
        fine_y_stop = (bounds_max[level, 1] << 1) + 3
        if fine_x_start < 0:
            fine_x_start = 0
        if fine_y_start < 0:
            fine_y_start = 0
        if fine_x_stop > fine_res:
            fine_x_stop = fine_res
        if fine_y_stop > fine_res:
            fine_y_stop = fine_res

        pair_y_start = fine_y_start >> 1
        pair_y_stop = (fine_y_stop + 1) >> 1

        for fine_x in range(fine_x_start, fine_x_stop):
            coarse_x0 = (fine_x - 1) >> 1
            coarse_x1 = coarse_x0 + 1
            weight_x1 = np.float32(0.75) if (fine_x & 1) == 0 else np.float32(0.25)
            weight_x0 = np.float32(1.0) - weight_x1
            if coarse_x0 < 0:
                coarse_x0 = 0
            if coarse_x1 > coarse_max:
                coarse_x1 = coarse_max

            x0_base = source_offset + coarse_x0 * coarse_res
            x1_base = source_offset + coarse_x1 * coarse_res
            output_base = destination_offset + fine_x * fine_res

            # Two adjacent fine y cells share the middle coarse sample.
            # Linear x interpolation is performed once per coarse y (left,
            # centre, right), then the two fine y outputs are derived with two
            # FMA-friendly operations.
            for coarse_y in range(pair_y_start, pair_y_stop):
                left_y = coarse_y - 1
                if left_y < 0:
                    left_y = 0
                right_y = coarse_y + 1
                if right_y > coarse_max:
                    right_y = coarse_max

                x_l = (
                    weight_x0 * coarse[x0_base + left_y]
                    + weight_x1 * coarse[x1_base + left_y]
                )
                x_c = (
                    weight_x0 * coarse[x0_base + coarse_y]
                    + weight_x1 * coarse[x1_base + coarse_y]
                )
                x_r = (
                    weight_x0 * coarse[x0_base + right_y]
                    + weight_x1 * coarse[x1_base + right_y]
                )

                fine_y = coarse_y << 1
                destination[output_base + fine_y] += x_c + np.float32(0.25) * (
                    x_l - x_c
                )
                destination[output_base + fine_y + 1] += x_c + np.float32(0.25) * (
                    x_r - x_c
                )

        # Bounds are only consumed by the next collapse. Level zero is final.
        if level > 1:
            if fine_x_start < bounds_min[level - 1, 0]:
                bounds_min[level - 1, 0] = fine_x_start
            if fine_y_start < bounds_min[level - 1, 1]:
                bounds_min[level - 1, 1] = fine_y_start
            if fine_x_stop - 1 > bounds_max[level - 1, 0]:
                bounds_max[level - 1, 0] = fine_x_stop - 1
            if fine_y_stop - 1 > bounds_max[level - 1, 1]:
                bounds_max[level - 1, 1] = fine_y_stop - 1


@njit(cache=True, fastmath=True, nogil=True)
def _build_hierarchy_layout(res: int, nlevels: int) -> tuple:
    """
    Build flattened hierarchy metadata.

    Computes the per-level resolution and flat-array offsets used by the scatter
    and collapse kernels.  Each level ``L`` has resolution ``res >> L``; level 0
    is the finest grid.

    Parameters
    ----------
    res : int
        Finest-grid resolution (cells per axis).
    nlevels : int
        Number of coarsening levels (deepest level has resolution ``res >> nlevels``).

    Returns
    -------
    level_offsets : np.ndarray[int64, shape=(nlevels+1,)]
        Cumulative flat-array start index for each level.
    level_resolutions : np.ndarray[int32, shape=(nlevels+1,)]
        Cell count per axis at each level.
    total_cells : int
        Total number of cells across all levels.
    """
    level_resolutions = np.empty(nlevels + 1, dtype=np.int32)
    level_offsets = np.empty(nlevels + 1, dtype=np.int64)
    total_cells = 0

    for level in range(nlevels + 1):
        level_res = res >> level
        level_resolutions[level] = level_res
        level_offsets[level] = total_cells
        total_cells += level_res * level_res

    return level_offsets, level_resolutions, total_cells


# ---------------------------------------------------------------------------
# Parallel scatter: split particles across the active Numba threads, run the
# full serial nested pipeline per chunk via prange, accumulate into a shared
# output grid.
# ---------------------------------------------------------------------------


@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _scatter_serial_chunk(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    m: NDArray[np.float32],
    h: NDArray[np.float32],
    res: int,
    box_x: np.float64,
    box_y: np.float64,
    ntarget: int,
    nlevels: int,
) -> np.ndarray:
    """
    Run the complete serial nested scatter pipeline on one particle chunk.

    JIT-callable wrapper around the full assign → scatter → collapse pipeline,
    designed to be invoked from ``prange`` inside the parallel kernel and as
    the single-thread entry point from the public API.  Each call builds its
    own private hierarchy and accumulates into a private finest grid.

    Parameters
    ----------
    x : np.ndarray[float64]
        Particle x-positions for this chunk, in [0, 1].
    y : np.ndarray[float64]
        Particle y-positions for this chunk, in [0, 1].
    m : np.ndarray[float32]
        Particle masses or weights for this chunk.
    h : np.ndarray[float32]
        Smoothing lengths for this chunk.
    res : int
        Finest-grid resolution (cells per axis).
    box_x : float64
        Periodic box size in x; 0.0 disables wrapping on that axis.
    box_y : float64
        Periodic box size in y; 0.0 disables wrapping on that axis.
    ntarget : int
        Target cells per kernel side at the assigned level.
    nlevels : int
        Maximum hierarchy depth.

    Returns
    -------
    np.ndarray[float32, shape=(res, res)]
        Partial pixel grid for this chunk.
    """
    level_index, active_nlevels_val = assign_levels(
        h, m, res, ntarget, nlevels, kernel_gamma
    )
    active_nlevels = np.int32(active_nlevels_val)

    level_offsets, level_resolutions, total_cells = _build_hierarchy_layout(
        res, active_nlevels
    )

    finest_cells = np.int64(res) * np.int64(res)
    finest = np.zeros(finest_cells, dtype=np.float32)
    n_coarse = total_cells - finest_cells
    coarse = np.zeros(
        n_coarse if n_coarse > np.int64(0) else np.int64(1), dtype=np.float32
    )

    bounds_min = np.empty((active_nlevels + 1, 2), dtype=np.int32)
    bounds_max = np.full((active_nlevels + 1, 2), np.int32(-1), dtype=np.int32)
    for level in range(active_nlevels + 1):
        lr = level_resolutions[level]
        bounds_min[level, 0] = lr
        bounds_min[level, 1] = lr

    _scatter_particles(
        x,
        y,
        m,
        h,
        level_index,
        level_offsets,
        level_resolutions,
        finest,
        coarse,
        box_x,
        box_y,
        bounds_min,
        bounds_max,
    )

    _collapse_serial_flat(
        finest,
        coarse,
        level_offsets,
        level_resolutions,
        active_nlevels,
        bounds_min,
        bounds_max,
    )
    return finest.reshape((res, res))


@njit(fastmath=True, parallel=True)
def _scatter_parallel_impl(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    m: NDArray[np.float32],
    h: NDArray[np.float32],
    res: int,
    box_x: np.float64 = np.float64(0.0),
    box_y: np.float64 = np.float64(0.0),
    ntarget: int = 6,
    nlevels: int = 4,
) -> np.ndarray:
    """
    Parallel nested multi-resolution scatter.

    Splits particles into one contiguous chunk per active Numba thread and
    calls the full serial nested pipeline on each chunk via ``prange``.  Each
    thread independently builds its own hierarchy and finest grid, avoiding
    any write-after-write races.  The per-thread grids are summed into the
    shared output after all chunks complete.

    Parameters
    ----------
    x : np.ndarray[float64]
        Particle x-positions; contiguous, in [0, 1].
    y : np.ndarray[float64]
        Particle y-positions; contiguous, in [0, 1].
    m : np.ndarray[float32]
        Contiguous particle masses or weights.
    h : np.ndarray[float32]
        Contiguous particle smoothing lengths.
    res : int
        Finest-grid resolution (cells per axis).
    box_x : float64
        Periodic box size in x; 0.0 disables wrapping on that axis.
    box_y : float64
        Periodic box size in y; 0.0 disables wrapping on that axis.
    ntarget : int
        Target cells per kernel side.
    nlevels : int
        Maximum hierarchy depth.

    Returns
    -------
    np.ndarray[float32, shape=(res, res)]
        Pixel grid of the projected quantity.

    See Also
    --------
    scatter : Serial implementation.
    """
    number_of_particles = x.size
    number_of_chunks = min(get_num_threads(), number_of_particles)

    output = np.zeros((res, res), dtype=np.float32)

    for chunk in prange(number_of_chunks):
        left_edge = chunk * number_of_particles // number_of_chunks
        right_edge = (chunk + 1) * number_of_particles // number_of_chunks

        output += _scatter_serial_chunk(
            x[left_edge:right_edge],
            y[left_edge:right_edge],
            m[left_edge:right_edge],
            h[left_edge:right_edge],
            res,
            box_x,
            box_y,
            ntarget,
            nlevels,
        )
    return output


def _prepare(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    m: NDArray[np.float32],
    h: NDArray[np.float32],
    res: int,
    ntarget: int,
    nlevels: int,
) -> tuple:
    """
    Validate all inputs and return typed, contiguous arrays with scalar config.

    Parameters
    ----------
    x : array-like
        Particle x-positions; must be 1-D with equal length.
    y : array-like
        Particle y-positions; must be 1-D with equal length.
    m : array-like
        Particle masses or weights; must be 1-D with equal length.
    h : array-like
        Particle smoothing lengths; must be 1-D with equal length.
    res : int-like
        Finest-grid resolution (cells per axis).
    ntarget : int-like
        Target cells per kernel side.
    nlevels : int-like
        Maximum hierarchy depth.

    Returns
    -------
    x : np.ndarray[float64]
        Contiguous x-position array.
    y : np.ndarray[float64]
        Contiguous y-position array.
    m : np.ndarray[float32]
        Contiguous mass array.
    h : np.ndarray[float32]
        Contiguous smoothing-length array.
    res : int
        Finest-grid resolution (cells per axis).
    ntarget : int
        Target cells per kernel side.
    nlevels : int
        Maximum hierarchy depth.
    """
    res, ntarget, nlevels = validate_hierarchy(res, ntarget, nlevels)
    x, y, m, h = prepare_particle_arrays({"x": x, "y": y}, m, h)
    return x, y, m, h, res, ntarget, nlevels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scatter(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    m: NDArray[np.float32],
    h: NDArray[np.float32],
    res: int,
    box_x: np.float64 = np.float64(0.0),
    box_y: np.float64 = np.float64(0.0),
    ntarget: int = 6,
    nlevels: int = 4,
) -> np.ndarray:
    """
    Create a weighted scatter plot using nested multi-resolution scatter.

    Particle inputs must be one-dimensional arrays with identical lengths.
    They are converted to contiguous float32/float64 arrays before entering Numba.

    Parameters
    ----------
    x : np.ndarray[np.float64]
        Array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.ndarray[np.float64]
        Array of y-positions of the particles. Must be bounded by [0, 1].

    m : np.ndarray[np.float32]
        Array of masses (or otherwise weights) of the particles.

    h : np.ndarray[np.float32]
        Array of smoothing lengths of the particles.

    res : int
        The number of pixels along one axis, i.e. this returns a square of
        ``res * res``. Must be divisible by ``2**nlevels``.

    box_x : np.float64
        Box size in x, in the same rescaled length units as x and y.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in x.

    box_y : np.float64
        Box size in y, in the same rescaled length units as x and y.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in y.

    ntarget : int
        Target number of pixels spanned by each kernel at its assigned level.
        Controls the accuracy/speed trade-off. Default is ``6``.

    nlevels : int
        Maximum number of coarsening levels in the hierarchy. Default is ``4``.

    Returns
    -------
    np.ndarray[np.float32, shape=(res, res)]
        Pixel grid of the projected quantity.

    See Also
    --------
    scatter_parallel : Parallel implementation of this function.
    """
    x, y, m, h, res, ntarget, nlevels = _prepare(x, y, m, h, res, ntarget, nlevels)
    return _scatter_serial_chunk(
        x, y, m, h, res, np.float64(box_x), np.float64(box_y), ntarget, nlevels
    )


def scatter_parallel(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    m: NDArray[np.float32],
    h: NDArray[np.float32],
    res: int,
    box_x: np.float64 = np.float64(0.0),
    box_y: np.float64 = np.float64(0.0),
    ntarget: int = 6,
    nlevels: int = 4,
) -> np.ndarray:
    """
    Create a weighted scatter plot using nested multi-resolution scatter in parallel.

    Uses the serial implementation when only one Numba thread is active, otherwise
    splits particles across threads and accumulates per-thread grids.

    Parameters
    ----------
    x : np.ndarray[np.float64]
        Array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.ndarray[np.float64]
        Array of y-positions of the particles. Must be bounded by [0, 1].

    m : np.ndarray[np.float32]
        Array of masses (or otherwise weights) of the particles.

    h : np.ndarray[np.float32]
        Array of smoothing lengths of the particles.

    res : int
        The number of pixels along one axis, i.e. this returns a square of
        ``res * res``. Must be divisible by ``2**nlevels``.

    box_x : np.float64
        Box size in x, in the same rescaled length units as x and y.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in x.

    box_y : np.float64
        Box size in y, in the same rescaled length units as x and y.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in y.

    ntarget : int
        Target number of pixels spanned by each kernel at its assigned level.
        Controls the accuracy/speed trade-off. Default is ``6``.

    nlevels : int
        Maximum number of coarsening levels in the hierarchy. Default is ``4``.

    Returns
    -------
    np.ndarray[np.float32, shape=(res, res)]
        Pixel grid of the projected quantity.

    See Also
    --------
    scatter : Serial implementation of this function.
    """
    x, y, m, h, res, ntarget, nlevels = _prepare(x, y, m, h, res, ntarget, nlevels)
    box_x = np.float64(box_x)
    box_y = np.float64(box_y)

    if get_num_threads() == 1:
        return _scatter_serial_chunk(x, y, m, h, res, box_x, box_y, ntarget, nlevels)

    return _scatter_parallel_impl(x, y, m, h, res, box_x, box_y, ntarget, nlevels)
