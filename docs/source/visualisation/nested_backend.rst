Nested multi-resolution backend
===============================

The ``nested`` backend is available for both 2D projections (see
:doc:`projection`) and 3D volume renders (see :doc:`volume_render`). It
follows the Sparse Multi-Scale Grid approach described in
`Benítez-Llambay (2025)`_: particles with large smoothing lengths are
deposited onto coarser grids, and the result is upsampled back to the
requested resolution afterwards.

.. _Benítez-Llambay (2025): https://iopscience.iop.org/article/10.3847/2515-5172/addab2

Motivation
----------

In the standard backends (``fast`` and friends for projections, ``scatter``
for volume renders) every particle visits every pixel or voxel within its
kernel compact-support radius. For a particle whose kernel spans :math:`N`
cells along each axis the cost is :math:`\mathcal{O}(N^2)` in 2D and
:math:`\mathcal{O}(N^3)` in 3D. Simulations with a large dynamic range in
smoothing length — zoom-in simulations, or cosmological volumes containing
both dense haloes and the diffuse IGM — contain particles whose kernels span
tens to hundreds of cells at typical resolutions. These few large particles
can dominate the total run time even though they are a tiny fraction of the
particle count, and the problem gets worse as the resolution increases.

The ``nested`` backend bounds the per-particle cost at
:math:`\mathcal{O}(n_\mathrm{target}^2)` in 2D and
:math:`\mathcal{O}(n_\mathrm{target}^3)` in 3D, regardless of smoothing
length.

Basic usage
-----------

Pass ``backend="nested"`` to any of the projection or volume-render
functions. The backend takes two optional arguments, ``ntarget`` and
``nlevels``, which can be passed straight to the high-level functions; any
extra keyword arguments are forwarded to the backend:

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.projection import project_gas
   from swiftsimio.visualisation.volume_render import render_gas

   data = load("cosmo_volume_example.hdf5")

   image = project_gas(
       data,
       resolution=1024,
       project="masses",
       parallel=True,
       backend="nested",
       ntarget=6,  # target cells per kernel diameter at each level
       nlevels=4,  # number of coarsening levels (requires resolution % 2**nlevels == 0)
   )

   grid = render_gas(
       data,
       resolution=256,
       project="masses",
       parallel=True,
       backend="nested",
   )

Both arguments default to ``ntarget=6`` and ``nlevels=4``. Everything else —
regions, depth-limited projections, rotations, masks and periodic wrapping —
works exactly as for the other backends.

Algorithm
---------

The pipeline has four stages. They are the same in 2D and 3D; only the
number of axes differs.

**1. Level assignment.**
For each particle, the number of finest-grid cells spanned by its kernel
compact-support radius is estimated:

.. math::

   N_\mathrm{cells} = \gamma_k \, h \, r

where :math:`r` is ``res`` and :math:`\gamma_k` is the Wendland-C2 kernel
gamma (:math:`1.897367` in 2D, :math:`1.936492` in 3D, the same values used
by the other backends). The particle is then assigned to hierarchy level
:math:`L`:

.. math::

   L = \mathrm{clip}\!\left(\left\lceil \log_2 \frac{N_\mathrm{cells}}{n_\mathrm{target}} \right\rceil,\; 0,\; L_\mathrm{max}\right)

Level 0 is the finest grid (resolution ``res``); level :math:`L` uses a grid
of resolution :math:`r / 2^L`. At its assigned level every kernel spans
approximately ``ntarget`` cells per axis. Particles with zero mass or a
negative smoothing length are skipped.

The hierarchy for ``res=1024``, ``nlevels=4`` (shown in 2D) looks like
this::

   Level 0 (finest)    1024 × 1024   — small, well-resolved kernels
   Level 1              512 ×  512
   Level 2              256 ×  256
   Level 3              128 ×  128
   Level 4 (coarsest)    64 ×   64   — very large, diffuse kernels

**2. Scatter.**
Each particle is deposited onto its assigned level using the Wendland-C2
kernel, the same kernel used by the standard backends. In 2D this is

.. math::

   W(r, H) = \frac{7}{\pi H^2} \left(1 - \frac{r}{H}\right)^4 \left(1 + \frac{4r}{H}\right),
   \qquad H = \gamma_k h.

Kernels smaller than half a cell at their level deposit their whole weight
into the cell that contains them. Periodic wrapping is handled as in the
other backends: a particle is deposited once for each periodic image whose
kernel overlaps ``[0, 1]``. Level-0 particles write directly into the output
grid; coarser particles write into a single flat array that stores all coarse
levels back-to-back.

**3. Collapse.**
Once every particle has been deposited, the hierarchy is collapsed
coarse-to-fine (from level :math:`L_\mathrm{max}` down to level 1). Each
coarse grid is upsampled by a factor of two (bilinearly in 2D, trilinearly in
3D) and added to the next finer grid. The upsampling stencil is::

   fine index 2i   draws 3/4 from coarse[i] and 1/4 from coarse[i-1]
   fine index 2i+1 draws 3/4 from coarse[i] and 1/4 from coarse[i+1]

applied independently along each axis. Only the bounding box of cells that
actually received a contribution is upsampled at each level, so the collapse
cost scales approximately with the number of particles rather than the grid
size.

**4. Return.**
The finest grid, which now contains the contributions from all levels, is
returned.

Accuracy
--------

The ``nested`` backend is an approximation. A particle at level
:math:`L > 0` is deposited onto a coarser grid and then spread over the fine
grid by upsampling, so the field it produces is the true kernel convolved
with the upsampling stencil. With the default ``ntarget=6`` individual cells
typically differ from the standard backend at the few-percent level, with
larger relative differences in faint cells near the edge of a kernel's
support, which carry very little of the total. Setting ``ntarget`` large
enough that every particle stays on level 0 reproduces the standard backend.

Neither the nested nor the standard backends renormalise their kernels, so
integrated quantities are conserved to about the same (percent-level)
accuracy in both; for projections, use ``renormalised`` or ``subsampled`` if
exact conservation matters more than speed.

Rendering the same particles with different region bounds or resolutions
changes their normalised positions in the last few bits of float64, and a
particle very close to a cell boundary can land in a different cell. This
affects a small fraction (a few percent) of cells.

Resolution constraint
---------------------

Because each level coarsens by exactly a factor of two, ``resolution`` must
be divisible by :math:`2^{\text{nlevels}}`. With the default ``nlevels=4``
this requires a multiple of 16 (e.g. 256, 512, 1024, 2048). An incompatible
combination raises a ``ValueError`` suggesting the nearest valid resolutions
and the largest ``nlevels`` that works with the requested one.

Memory usage
------------

The backend allocates the output grid (:math:`r^d` float32 values, with
:math:`d` = 2 or 3) plus a flat array holding all coarse levels, of total
size

.. math::

   \sum_{L=1}^{L_\mathrm{max}} (r / 2^L)^d
   < \frac{r^d}{2^d - 1},

so the footprint is less than :math:`4/3` of the output image in 2D and
:math:`8/7` of the output grid in 3D, independent of ``nlevels``. Only levels
that are actually occupied are allocated. In the parallel implementations
each thread holds a private copy of this allocation, so peak memory scales
with the number of threads, as for the other parallel backends.

Performance
-----------

The speedup depends on how many particles have kernels much larger than
``ntarget`` cells. For 200,000 particles with smoothing lengths spread
log-uniformly over nearly three decades, projected at ``resolution=1024``
with 32 threads, ``nested`` took 0.12 s against 1.2 s for ``fast``. For data
where all kernels are only a few cells across, every particle stays on level
0 and the nested and standard backends run at similar speeds.

When to use it
--------------

Use ``nested`` when:

* The simulation has a wide dynamic range in smoothing length (zoom-in runs,
  cosmological boxes including the IGM, multi-phase ISM models).
* You are working at high resolution, where large kernels span many cells.
* A few-percent approximation in individual cells is acceptable.

Prefer a standard backend when:

* All particles have similar smoothing lengths.
* You need exact per-cell agreement with the reference kernel, or (for
  projections) converged results at low resolution (``subsampled``).
* The resolution cannot be made divisible by ``2**nlevels``.
* Memory per thread is a constraint and the thread count is high.

Choosing ``ntarget`` and ``nlevels``
------------------------------------

``ntarget`` controls how many cells each kernel spans at its assigned level.
Higher values keep particles on finer grids (less upsampling, higher
accuracy) at the cost of more kernel evaluations per particle. The default of
``6`` is a good all-round choice; values of 4–10 cover the practical
speed/accuracy trade-off.

``nlevels`` sets the depth of the hierarchy. More levels let very large
kernels be placed on very coarse grids, giving bigger speedups for extremely
diffuse particles, at the cost of a stricter divisibility requirement on
``resolution``. The default of ``4`` is enough for most smoothing-length
distributions.

.. code-block:: python

   # Higher accuracy: fewer particles pushed to coarse levels.
   accurate = project_gas(data, resolution=2048, backend="nested", ntarget=10)

   # Maximum speed: deeper hierarchy, smaller ntarget.
   # Requires resolution divisible by 2**5 = 32.
   quick = render_gas(data, resolution=512, backend="nested", ntarget=4, nlevels=5)
