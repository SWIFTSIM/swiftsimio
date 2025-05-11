	Masking
=======

:mod:`swiftsimio` provides unique functionality (when compared to other
software packages that read HDF5 data) through its masking facility.

SWIFT snapshots contain cell metadata that allow us to spatially mask the
data ahead of time. :mod:`swiftsimio` provides a number of objects that help
with this. This functionality is provided through the :mod:`swiftsimio.masks`
sub-module but is available easily through the :meth:`swiftsimio.mask`
top-level function.

This functionality is used heavily in `swiftgalaxy`_.

There are two types of mask, with the default only allowing spatial masking.
Full masks require significantly more memory overhead and are generally much
slower than the spatial only mask.

.. _`swiftgalaxy`: https://github.com/SWIFTSIM/swiftgalaxy

Spatial-only masking
--------------------

Spatial only masking is approximate and allows you to only load particles
within a given region. It is precise to the top-level cells that are defined
within SWIFT. It will always load all of the particles that you request, but
for simplicity it may also load some particles that are slightly outside
of the region of interest. This is because it works as follows:

1. Load the top-level cell metadata.
2. Find the overlap between the specified region and these cells.
3. Load all cells within that overlap.

As you can see, the edges of regions may load in extra information as we
always load the whole top-level cell.

Example
^^^^^^^

In this example we will use the :obj:`swiftsimio.masks.SWIFTMask` object
to load the the octant of the box closes to the origin.

.. code-block:: python

   import swiftsimio as sw

   filename = "cosmo_volume_example.hdf5"

   mask = sw.mask(filename)
   # The full metadata object is available from within the mask
   boxsize = mask.metadata.boxsize
   # load_region is a 3x2 list [[left, right], [bottom, top], [front, back]]
   load_region = [[0.0 * b, 0.5 * b] for b in boxsize]

   # Constrain the mask
   mask.constrain_spatial(load_region)

   # Now load the snapshot with this mask
   data = sw.load(filename, mask=mask)

``data`` is now a regular :obj:`swiftsimio.reader.SWIFTDataset` object, but
it only ever loads particles that are (approximately) inside the
``load_region`` region.

Importantly, this method has a tiny memory overhead, and should also have a
relatively small overhead when reading the data. This allows you to use snapshots
that are much larger than the available memory on your machine and process them
with ease.

It is also possible to build up a region with a more complicated geometry by
making repeated calls to :meth:`~swiftsimio.masks.SWIFTMask.constrain_spatial`
and setting the optional argument ``intersect=True``. By default any existing
selection of cells would be overwritten; this option adds any additional cells
that need to be selected for the new region to the existing selection instead.
For instance, to add the diagonally opposed octant to the selection made above
(and so obtain a region shaped like two cubes with a single corner touching):

.. code-block:: python

   additional_region = [[0.5 * b, 1.0 * b] for b in boxsize]
   mask.constrain_spatial(additional_region, intersect=True)

In the first call to :meth:`~swiftsimio.masks.SWIFTMask.constrain_spatial` the
``intersect`` argument can be set to ``True`` or left ``False`` (the default): since
no mask yet exists both give the same result.

Periodic boundaries
^^^^^^^^^^^^^^^^^^^

The mask region is aware of the periodic box boundaries. Let's take for example a
region shaped like a "slab" in the :math:`x-y` plane with :math:`|z|<0.1L_\mathrm{box}`.
One way to write this is by thinking of the :math:`z<0` part as
lying at the upper edge of the box:

.. code-block:: python

   mask = sw.mask(filename)
   mask.constrain_spatial(
       [
           None,
           None,
           [0.0 * mask.metadata.boxsize[2], 0.1 * mask.metadata.boxsize[2]],
       ]
   )
   mask.constrain_spatial(
       [
           None,
           None,
           [0.9 * mask.metadata.boxsize[2], 1.0 * mask.metadata.boxsize[2]],
       ],
       intersect=True,
   )

This is a bit inconvenient though since the region is actually contiguous if we
account for the periodic boundary. :meth:`~swiftsimio.masks.SWIFTMask.constrain_spatial` allows us
to select a region straddling the periodic boundary, for example this is an
equivalent selection:

.. code-block:: python

   mask = sw.mask(filename)
   mask.constrain_spatial(
       [
           None,
	   None,
	   [-0.1 * mask.metadata.boxsize[2], 0.1 * mask.metadata.boxsize[2]],
       ]
   )

Note that masking never result in periodic copies of particles, nor does it shift
particle coordinates to match the region defined; particle coordinates always
lie in the range :math:`[-L_\mathrm{box}, L_\mathrm{box}]`. For example reading
a region that extends beyond the box in all directions produces exactly one copy
of every particle and is equivalent to providing no spatial mask:

.. code-block:: python

   mask = sw.mask(filename)
   mask.constrain_spatial(
       [
           None,
	   None,
	   [-0.1 * mask.metadata.boxsize[2], 1.1 * mask.metadata.boxsize[2]],
       ]
   )

Remember to wrap the coordinates yourself if relevant! Alternatively, the
`swiftgalaxy`_ package offers support for coordinate transformations including
periodic boundaries.

Another equivalent region for the :math:`|z|<0.1L_\mathrm{box}` slab can be written
by setting the lower bound to a greater value than the upper bound, the code will
interpret this as a request to start at the lower bound, wrap through the upper
periodic boundary and continue until the (numerically lower value of) the upper
bound is reached:

.. code-block:: python

   mask = sw.mask(filename)
   mask.constrain_spatial(
       [
           None,
	   None,
	   [0.9 * mask.metadata.boxsize[2], 0.1 * mask.metadata.boxsize[2]],
       ]
   )

The coordinates defining the region must always be in the interval
:math:`[-0.5L_\mathrm{box}, 1.5L_\mathrm{box}]`. This allows enough flexibility to
define all possible regions.

Implementation details
^^^^^^^^^^^^^^^^^^^^^^

SWIFT snapshots group particles according to the cell that they occupy so that
particles belonging to a cell are stored contiguously. The cells form a regular grid
covering the simulation domain. However, SWIFT does not guarantee that all particles
that belong to a cell are within the boundaries of a cell at the time when a snapshot
is produced (particles are moved between cells at intervals, but may drift outside of
their current cell before being re-assigned). Snapshots contain metadata defining
the "bounding box" of each cell that contains all particles assigned to it at the
time that the snapshot was written. :mod:`swiftsimio` uses this information when
deciding what cells to read, so you may find that the "extra" particles read in
outside of the explicitly asked for have an irregular boundary with cuboid protrusions
or indentations. This is normal: the cells read in are exactly those needed to
guarantee that all particles in the specified region of interest are captured. It is
therefore advantageous to make the region as small and tightly fit to the analysis
task as possible - in particular, trying to align it with the cell boundaries will
typically result in an I/O overhead as neighbouring cells with particles that have
drifted into the region are read in. Unless these particles are actually needed, it
is actually better for performance to *avoid* the cell boundaries when defining the
region.

Older SWIFT snapshots lack the metadata to know exactly how far particles have
drifted out of their cells. In ``v10.2.0`` or newer, if :mod:`swiftsimio` does not
find this metadata, it will pad the region (by 0.2 times the cell length by default).

.. warning::

   In the worst case that the region consists of one cell and the padding extends to all
neighbouring cells, this can result in up to a factor of :math:`3^3=27` additional
I/O overhead. Older :mod:`swiftsimio` versions instead risk missing particles near
the region boundary.

In the unlikely case that particles drift more than 0.2 times
the cell length away from their "home" cell and the cell bounding-box metadata is not
present, some particles can be missed when applying a spatial mask. The padding of
the region can be extended or switched off with the ``safe_padding`` parameter:

.. code-block:: python

   mask = sw.mask(filename)
   lbox = mask.metadata.boxsize
   mask.constrain_spatial(
       [[0.4 * lbox, 0.6 * lbox] for lbox in mask.metadata.boxsize],
       safe_padding=False,  # padding switched off
   )
   mask.constrain_spatial(
       [[0.4 * lbox, 0.6 * lbox] for lbox in mask.metadata.boxsize],
       safe_padding=0.5,  # pad more, by 0.5 instead of 0.2 cell lengths
   )


Full mask
---------

The below example shows the use of a full masking object, used to constrain
densities of particles and only load particles within that density window.

.. code-block:: python
   
   import swiftsimio as sw

   # This creates and sets up the masking object.
   mask = sw.mask("cosmological_volume.hdf5", spatial_only=False)

   # This ahead-of-time creates a spatial mask based on the cell metadata.
   mask.constrain_spatial([
       [0.2 * mask.metadata.boxsize[0], 0.7 * mask.metadata.boxsize[0]],
       None,
       None]
   )

   # Now, just for fun, we also constrain the density between
   # 0.4 g/cm^3 and 0.8. This reads in the relevant data in the region,
   # and tests it element-by-element. Note that using masks of this type
   # is significantly slower than using the spatial-only masking.
   density_units = mask.units.mass / mask.units.length**3
   mask.constrain_mask("gas", "density", 0.4 * density_units, 0.8 * density_units)

   # Now we can grab the actual data object. This includes the mask as a parameter.
   data = sw.load("cosmo_volume_example.hdf5", mask=mask)


When the attributes of this data object are accessed, *only* the ones that
belong to the masked region (in both density and spatial) are read. I.e. if I
ask for the temperature of particles, it will recieve an array containing
temperatures of particles that lie in the region [0.2, 0.7] and have a
density between 0.4 and 0.8 g/cm^3.

Row Masking
-----------

For certian scenarios, in particular halo catalogues, all arrays are of the
same length (you can check this through the ``metadata.homogeneous_arrays``
attribute). Often, you are interested in a handful of, or a single, row,
corresponding to the properties of a particular object. You can use the
methods ``constrain_index`` and ``constrain_indices`` to do this, which
return ``swiftsimio`` data objects containing arrays with only those
rows.

.. code-block:: python
    
    import swiftsimio as sw

    mask = sw.mask(filename)

    mask.constrain_indices([1, 99, 23421])

    data = sw.load(filename, mask=mask)

Here, the length of all the arrays will be 3. A quick performance note: if you
are using many indices (over 1000), you will want to set ``spatial_only=False``
to potentially benefit from range reading of overlapping rows in a single chunk.

Writing subset of snapshot
--------------------------
In some cases it may be useful to write a subset of an existing snapshot to its
own hdf5 file. This could be used, for example, to extract a galaxy halo that 
is of interest from a snapshot so that the file is easier to work with and transport.

To do this the ``write_subset`` function is provided. It can be used, for example,
as follows

.. code-block:: python

    import swiftsimio as sw                                                 
    import unyt                                                             
    
    mask = sw.mask("eagle_snapshot.hdf5")                                       
    mask.constrain_spatial([
        [unyt.unyt_quantity(100, unyt.kpc), unyt.unyt_quantity(1000, unyt.kpc)], 
        None, 
        None])                                   
    
    sw.subset_writer.write_subset("test_subset.hdf5", mask)

This will write a snapshot which contains the particles from the specified snapshot 
whose :math:`x`-coordinate is within the range [100, 1000] kpc. This function uses the 
cell mask which encompases the specified spatial domain to successively read portions 
of datasets from the input file and writes them to a new snapshot. 

Due to the coarse grained nature of the cell mask, particles from outside this range 
may also be included if they are within the same top level cells as particles that 
fall within the given range.

Please note that it is important to run ``constrain_spatial`` as this generates
and stores the cell mask needed to write the snapshot subset. 
