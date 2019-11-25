Masking
=======

:mod:`swiftsimio` provides unique functionality (when compared to other
software packages that read HDF5 data) through its masking facility.

SWIFT snapshots contain cell metadata that allow us to spatially mask the
data ahead of time. :mod:`swiftsimio` provides a number of objects that help
with this. This functionality is provided through the :mod:`swiftsimio.masks`
sub-module but is available easily through the :meth:`swiftsimio.mask`
top-level function.

Note that masking does not currently work with simulations that use
dithering.

This functionality is used heavily in our `VELOCIraptor integration library`_
for only reading data that is near bound objects.

There are two types of mask, with the default only allowing spatial masking.
Full masks require significantly more memory overhead and are generally much
slower than the spatial only mask.

.. _`VELOCIraptor integration library`: https://github.com/swiftsim/velociraptor-python

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
to load the bottom left 'half' corner of the box.

.. code-block:: python

   import swiftsimio as sw

   filename = "cosmological_volume.hdf5"

   mask = sw.mask(filename)
   # The full metadata object is available from within the mask
   boxsize = mask.metadata.boxsize
   # load_region is a 3x2 list [[left, right], [bottom, top], [front, back]]
   load_region = [[0.0 * b, 0.5 * b] for b in boxsize]

   # Constrain the mask
   mask.constrain_spatial(load_region)

   # Now load the snapshot with this mask
   data = load(filename, mask=mask)

``data`` is now a regular :obj:`swiftsimio.reader.SWIFTDataset` object, but
it only ever loads particles that are (approximately) inside the
``load_region`` region.

Importantly, this method has a tiny memory overhead, and should also have a
relatively small overhead when reading the data. This allows you to use snapshots
that are much larger than the available memory on your machine and process them
with ease.

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
   data = sw.load("cosmological_volume.hdf5", mask=mask)


When the attributes of this data object are accessed, *only* the ones that
belong to the masked region (in both density and spatial) are read. I.e. if I
ask for the temperature of particles, it will recieve an array containing
temperatures of particles that lie in the region [0.2, 0.7] and have a
density between 0.4 and 0.8 g/cm^3.
