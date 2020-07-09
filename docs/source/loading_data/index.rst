Loading Data
============

The main purpose of :mod:`swiftsimio` is to load data. This section will tell
you all about four main objects:

+ :obj:`swiftsimio.reader.SWIFTUnits`, responsible for creating a correspondence between
  the SWIFT units and :mod:`unyt` objects.
+ :obj:`swiftsimio.reader.SWIFTMetadata`, responsible for loading any required information
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
+ ``data.metadata``, an instance of :obj:`swiftsimio.reader.SWIFTMetadata`
+ ``data.units``, an instance of :obj:`swiftsimio.reader.SWIFTUnits`

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


Reading particle data
---------------------

To find out what particle properties are present in our snapshot, we can use
the instance of :obj:`swiftsimio.reader.SWIFTMetadata`, ``data.metadata``,
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


Non-unyt properties
-------------------

Each data array has some custom properties that are not present within the base
:obj:`unyt.unyt_array` class. We create our own version of this in
:obj:`swiftsimio.objects.cosmo_array`, which allows each dataset to contain
its own cosmology and name properties.

For instance, should you ever need to know what a dataset represents, you can
ask for a description:

.. code-block:: python

   print(rho_gas.name)

which will output ``Co-moving mass densities of the particles``. They include
scale-factor information, too, through the ``cosmo_factor`` object,

.. code-block:: python

   # Conversion factor to make the densities a physical quantity
   print(rho_gas.cosmo_factor.a_factor)
   physical_rho_gas = rho_gas.cosmo_factor.a_factor * rho_gas

   # Symbolic scale-factor expression
   print(rho_gas.cosmo_factor.expr)

which will output ``132651.002785671`` and ``a**(-3.0)``. This is an easy way
to convert your co-moving values to physical ones.

An even easier way to convert your properties to physical is to use the
built-in ``to_physical`` and ``convert_to_physical`` methods, as follows:

.. code-block:: python

   physical_rho_gas = rho_gas.to_physical()

   # Convert in-place
   rho_gas.convert_to_physical()


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
