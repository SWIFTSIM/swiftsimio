.. _loading-data:

Loading Data
============

The main purpose of :mod:`swiftsimio` is to load data. This section will tell
you all about four main objects:

+ :obj:`swiftsimio.metadata.objects.SWIFTUnits`, responsible for creating a correspondence between
  the SWIFT units and :mod:`unyt` objects.
+ :obj:`swiftsimio.metadata.objects.SWIFTMetadata`, responsible for loading any required information
  from the SWIFT headers into python-readable data.
+ :obj:`swiftsimio.reader.SWIFTDataset`, responsible for holding all particle data, and
  keeping track of the above two objects.
+ :obj:`swiftsimio.reader.SWIFTParticleTypeMetadata`, responsible for
  cataloguing metadata just about individual particle types, like gas,
  including what particle fields are present.


To get started, first locate any SWIFT data that you wish to analyse. If you
don't have any handy, you can always download our test cosmological volume
at:

``http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/cosmo_volume_example.hdf5``

with associated halo catalogue at

``http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/cosmo_volume_example.properties``

which is needed should you wish to use the ``velociraptor`` integration library in the
visualisation examples.

To create your first instance of :obj:`swiftsimio.reader.SWIFTDataset`, you can
use the helper function :mod:`swiftsimio.load` as follows:

.. code-block:: python

   from swiftsimio import load

   # Of course, replace this path with your own snapshot should you be using
   # custom data.
   data = load("cosmo_volume_example.hdf5")

The type of ``data`` is now :obj:`swiftsimio.reader.SWIFTDataset`. Have a
quick look around this dataset in an ``iPython`` shell, or a ``jupyter``
notebook, and you will see that it contains several sub-objects:

+ ``data.gas``, which contains all information about gas particles in the
  simulation.
+ ``data.dark_matter``, likewise containing information about the dark matter
  particles in the simulation.
+ ``data.metadata``, an instance of :obj:`swiftsimio.metadata.objects.SWIFTSnapshotMetadata`
+ ``data.units``, an instance of :obj:`swiftsimio.metadata.objects.SWIFTUnits`

Using metadata
--------------

Let's begin by reading some useful metadata straight out of our
``data.metadata`` object. For instance, we may want to know the box-size of
our simulation:

.. code-block:: python

   meta = data.metadata
   boxsize = meta.boxsize

   print(boxsize)

This will output ``[142.24751067 142.24751067 142.24751067] Mpc`` - note
the units that are attached. These units being attached to everything is one
of the key advantages of using :mod:`swiftsimio`. It is really easy to convert
between units; for instance if we want that box-size in kiloparsecs,

.. code-block:: python

   boxsize.convert_to_units("kpc")

   print(boxsize)

Now outputting ``[142247.5106242 142247.5106242 142247.5106242] kpc``. Neat!
This is all thanks to our tight integration with :mod:`unyt`. If you have more
complex units, it is often useful to specify them in terms of :mod:`unyt`
objects as follows:

.. code-block:: python

   import unyt

   new_units = unyt.cm * unyt.Mpc / unyt.kpc
   new_units.simplify()

   boxsize.convert_to_units(new_units)

In general, we suggest using :mod:`unyt` unit objects rather than strings. You
can find more information about :mod:`unyt` on the `unyt documentation website`_.

.. _`unyt documentation website`: https://unyt.readthedocs.io/en/stable/

There is lots of metadata available through this object; the best way to see
this is by exploring the object using ``dir()`` in an interactive shell, but
as a summary:

+ All metadata from the snapshot is available through many variables, for example
  the ``meta.hydro_scheme`` property.
+ The numbers of particles of different types are available through
  ``meta.n_{gas,dark_matter,stars,black_holes}``.
+ Several pre-LaTeXed strings are available describing the configuration state
  of the code, such as ``meta.hydro_info``, ``meta.compiler_info``.
+ Several snapshot-wide parameters, such as ``meta.a`` (current scale factor),
  ``meta.t`` (current time), ``meta.z`` (current redshift), ``meta.run_name``
  (the name of this run, specified in the SWIFT parameter file), and
  ``meta.snapshot_date`` (a :mod:`datetime` object describing when the
  snapshot was written to disk).
+ If you have ``astropy`` installed, you can also use the ``metadata.cosmology``
  object, which is an ``astropy.cosmology.w0waCDM`` instance.


Reading particle data
---------------------

To find out what particle properties are present in our snapshot, we can print
the available particle types. For example:

.. code-block:: python

   data
   
prints the available particle types (or, more generally, groups):

.. code-block:: python

   SWIFT dataset at cosmo_volume_example.hdf5.
   Available groups: gas, dark_matter, stars, black_holes

The properties available for a particle type can be similarly printed:

.. code-block:: python

   data.dark_matter

gives:

.. code-block:: python

   SWIFT dataset at cosmo_volume_example.hdf5.
   Available fields: coordinates, masses, particle_ids, velocities
   
With compatible python interpreters, the available fields (and other attributes
such as functions) can be seen using the tab completion feature, for example
typing `>>> data.dark_matter.` at the command prompt and pressing tab twice
gives:

.. code-block:: python

   data.dark_matter.coordinates                  data.dark_matter.masses
   data.dark_matter.filename                     data.dark_matter.metadata
   data.dark_matter.particle_ids                 data.dark_matter.generate_empty_properties()
   data.dark_matter.group                        data.dark_matter.units
   data.dark_matter.group_metadata               data.dark_matter.velocities
   data.dark_matter.group_name                   

The available fields can also be accessed programatically using the instance of
:obj:`swiftsimio.reader.SWIFTMetadata`, ``data.metadata``,
which contains several instances of
:obj:`swiftsimio.reader.SWIFTParticleTypeMetadata` describing what kinds of
fields are present in gas or dark matter:

.. code-block:: python

   # Note that gas_properties is an instance of SWIFTParticleTypeMetadata
   print(data.metadata.gas_properties.field_names)

This will print a large list, like

.. code-block:: python

   ['coordinates',
   'densities',
   ...
   'temperatures',
   'velocities']

These individual attributes can be accessed through the object-oriented
interface, for instance,

.. code-block:: python

   x_gas = data.gas.coordinates
   rho_gas = data.gas.densities
   x_dm = data.dark_matter.coordinates

Only at this point is any information actually read from the snapshot, so far
we have only read three arrays into memory - in this case corresponding to
``/PartType0/Coordinates``, ``/PartType1/Coordinates``, and
``/PartType0/Densities``.

This allows you to be quite lazy when writing scripts; you do not have to
write, for instance, a block at the start of the file with a
``with h5py.File(...) as handle:`` and read all of the data at once, you can
simply access data whenever you need it through this predictable interface.

Just like the boxsize, these carry symbolic :mod:`unyt` units,

.. code-block:: python

   print(x_gas.units)

will output ``Mpc``. We can again convert these to whatever units
we like. For instance, should we wish to convert our gas densities to solar
masses per cubic megaparsec,

.. code-block:: python

   new_density_units = unyt.Solar_Mass / unyt.Mpc**3

   rho_gas.convert_to_units(new_density_units)

   print(rho_gas.units.latex_repr)

which will output ``'\\frac{M_\\odot}{\\rm{Mpc}^{3}}'``. This is a LaTeX
representation of those symbolic units that we just converted our data to -
this is very useful for making plots as it can ensure that your data and axes
labels always have consistent units.


Named columns
-------------

SWIFT can output custom metadata in ``SubgridScheme/NamedColumns`` for multi
dimensional tables containing columns that carry individual data. One common
example of this is the element mass fractions of gas and stellar particles.
These are then placed in an object hierarchy, as follows:

.. code-block:: python

   print(data.gas.element_mass_fractions)


This will output: Named columns instance with ['hydrogen', 'helium',
'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
available for "Fractions of the particles' masses that are in the given
element"

Then, to access individual columns (in this case element abundances):

.. code-block:: python

   # Access the silicon abundance
   data.gas.element_mass_fractions.silicon


User-defined particle types
---------------------------

It is now possible to add user-defined particle types that are not already
present in the :mod:`swiftsimio` metadata. All you need to do is specify the
three names (see below) and then the particle datasets that you have provided
in SWIFT will be automatically read.

.. code-block:: python

   import swiftsimio as sw
   import swiftsimio.metadata.particle as swp
   from swiftsimio.objects import cosmo_factor, a

   swp.particle_name_underscores[6] = "extratype"
   swp.particle_name_class[6] = "Extratype"
   swp.particle_name_text[6] = "Extratype"

   data = sw.load(
       "extra_test.hdf5",
   )


Reading from an open file
-------------------------

:mod:`swiftsimio` normally opens and closes the HDF5 snapshot file for
each operation. This is convenient for interactive use and avoids
leaving files open for long periods of time, but sometimes it might be
desirable to minimize the number of file open and close operations.

It is possible to pass an open :obj:`h5py.File` object to
:mod:`swiftsimio.load` and :mod:`swiftsimio.mask` in place of the
filename. In this case swiftsimio will do all file access through the
provided file object. This allows us to read multiple datasets while
only opening and closing the file once. For example:

.. code-block:: python

   import h5py
   import swiftsimio as sw

   with h5py.File("cosmo_volume_example.hdf5","r") as snap_file:
      data = sw.load(snap_file)
      pos = data.dark_matter.coordinates
      vel = data.dark_matter.velocities
      ids = data.dark_matter.particle_ids

This would open the snapshot file, read the dark matter particle
positions, velocities and IDs, then close the file.


Reading from a remote file
--------------------------

:mod:`swiftsimio` is able to read from snapshots hosted on a remote
server using the `hdfstream
<https://hdfstream-python.readthedocs.io/en/latest>`_ python
module. This is useful if you're interested in accessing a small part
of a larger snapshot: you can read a small region or a subset of
particle properties without downloading the whole snapshot.

To open a remote snapshot, you can pass a :obj:`hdfstream.RemoteFile`
object to :mod:`swiftsimio.load` and :mod:`swiftsimio.mask` in place
of the filename. For example, you can open one of the SWIFT example
snapshots with:

.. code-block:: python

   import hdfstream
   from swiftsimio import load

   snap_file = hdfstream.open("cosma", "Tests/SWIFT/IOExamples/ssio_ci_04_2025/EagleSingle.hdf5")
   data = load(snap_file)

Here, ``data`` will be a :obj:`swiftsimio.reader.SWIFTDataset`. It
functions in the same way as described in the :ref:`loading-data`
section above, except that instead of reading data from a local HDF5
file, it requests data from the server.

Opening a snapshot like this only downloads a small amount of
metadata. Accessing particle properties, such as coordinates, will
trigger another download:

.. code-block:: python

   pos = data.dark_matter.coordinates

This will download the dark matter particle coordinates and return an
array with units and cosmological factors attached.

To read part of a remote snapshot, we can use swiftsimio's
:ref:`masking` feature as we would with a local snapshot, but passing
the remote file to :mod:`swiftsimio.mask` :mod:`swiftsimio.load` in
place of the filename.

.. code-block:: python

   import hdfstream
   import swiftsimio as sw

   snap_file = hdfstream.open("cosma", "Tests/SWIFT/IOExamples/ssio_ci_04_2025/EagleSingle.hdf5")

   mask = sw.mask(snap_file)
   # The full metadata object is available from within the mask
   boxsize = mask.metadata.boxsize
   # load_region is a 3x2 list [[left, right], [bottom, top], [front, back]]
   load_region = [[0.0 * b, 0.5 * b] for b in boxsize]

   # Constrain the mask
   mask.constrain_spatial(load_region)

   # Now load the snapshot with this mask
   data = sw.load(snap_file, mask=mask)
