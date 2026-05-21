The ``cosmo_array``
===================

:mod:`swiftsimio` uses a customized class based on the :class:`~unyt.array.unyt_array`
to store data arrays. The :class:`~swiftsimio.objects.cosmo_array` has all of the same
functionality as the :class:`~unyt.array.unyt_array`, but also adds information about
data transformation between physical and comoving coordinates, and descriptive metadata.

For instance, should you ever need to know what a dataset represents, you can
ask for a description by accessing the ``name`` attribute:

.. code-block:: python

   print(rho_gas.name)

which will output ``Co-moving mass densities of the particles``.

The cosmology information is stored in three attributes:

 + ``comoving``
 + ``cosmo_factor``
 + ``valid_transform``

The ``comoving`` attribute specifies whether the array is a physical (``True``) or
comoving (``False``) quantity, while the ``cosmo_factor`` stores the expression needed
to convert back and forth between comoving and physical quantities and the value of
the scale factor. The conversion factors can be accessed like this:

.. code-block:: python

   # Conversion factor to make the densities a physical quantity
   print(rho_gas.cosmo_factor.a_factor)
   physical_rho_gas = rho_gas.cosmo_factor.a_factor * rho_gas

   # Symbolic scale-factor expression
   print(rho_gas.cosmo_factor.expr)

which will output ``132651.002785671`` and ``a**(-3.0)``. Converting an array to/from
physical/comoving is done with the :meth:`~swiftsimio.objects.cosmo_array.to_physical`,
:meth:`~swiftsimio.objects.cosmo_array.to_comoving`,
:meth:`~swiftsimio.objects.cosmo_array.convert_to_physical` and
:meth:`~swiftsimio.objects.cosmo_array.to_comoving` methods. These can also convert units
at the same time, for instance:

.. code-block:: python

   import unyt as u

   # Create copies:
   physical_rho_gas = rho_gas.to_physical()  # preserves current units
   physical_rho_gas_cgs = rho_gas.to_physical(u.g / u.cm**3)

   # Convert in-place:
   rho_gas.convert_to_physical()  # preserves current units
   rho_gas.convert_to_physical(u.g / u.cm ** 2)

Equivalently, the :meth:`~swiftsimio.objects.cosmo_array.to` can accept a comoving
argument:

.. code-block:: python

   # Create copies:
   physical_rho_gas = rho_gas.to(comoving=False)  # preserves current unit
   physical_rho_gas_cgs = rho_gas.to(u.g / u.cm**3, comoving=False)

If you instead want to get the raw array values in either physical or comoving units, the
:meth:`~swiftsimio.objects.cosmo_array.to_physical_value` and
:meth:`~swiftsimio.objects.cosmo_array.to_comoving_value` are provided, analogous to the
:meth:`~unyt.unyt_array.unyt_array.to_value` (which also has a ``comoving`` argument
available):

.. code-block:: python

   physical_rho_gas_cgs = rho_gas.to_physical_value(u.g / u.cm**3)
   comoving_rho_gas_cgs = rho_gas.to_comoving_value(u.g / u.cm**3)
   # Or:
   physical_rho_gas_cgs = rho_gas.to_value(u.g / u.cm**3, comoving=False)

The ``valid_transform`` is a boolean flag that is set to ``False`` for some arrays that
don't make sense to convert to comoving.

:class:`~swiftsimio.objects.cosmo_array` supports array arithmetic and the entire
:mod:`numpy` range of functions. Attempting to combine arrays (e.g. by addition) will
validate the cosmology information first. The implementation is designed to be permissive:
it will only raise exceptions when a genuinely invalid combination is encountered, but is
tolerant of missing cosmology information. When one argument in a relevant operation (like
addition, for example) is not a :class:`~swiftsimio.objects.cosmo_array` the attributes of
the :class:`~swiftsimio.objects.cosmo_array` will be assumed for both arguments. In such
cases a warning is produced stating that this assumption has been made.

.. note::

   :class:`~swiftsimio.objects.cosmo_array` and the related
   :class:`~swiftsimio.objects.cosmo_quantity` are now intended to support all
   :mod:`numpy` functions, propagating units and cosmology information correctly through
   mathematical operations. Try making a histogram with weights and ``density=True`` with
   :func:`numpy.histogram`! There are a large number of functions and a very large number
   of possible parameter combinations, so some corner cases may have been missed in
   development. Please report any errors or unexpected results using github issues or
   other channels so that they can be fixed. Currently :mod:`scipy` functions are not
   supported (although some might "just work"). Requests to support specific functions can
   be accommodated.

Creating "cosmo" arrays and scalars
-----------------------------------

To make the most of the utility offered by the :class:`~swiftsimio.objects.cosmo_array`
class, it is helpful to know how to create your own. A good template for this looks like:

.. code-block:: python

   import unyt as u
   from swiftsimio.objects import cosmo_array, cosmo_factor

   # suppose the scale factor is 0.5 and it scales as a**1, then:
   my_cosmo_array = cosmo_array(
       [1, 2, 3],
       u.Mpc,
       comoving=True,
       scale_factor=0.5,  # a=0.5, i.e. z=1
       scale_exponent=1,  # distances scale as a**1, so the scale exponent is 1
   )
   # consider getting the scale factor from metadata when applicable, i.e. replace:
   # scale_factor=0.5
   # with:
   # scale_factor=data.metadata.scale_factor

This can be long to write out. To alleviate this a helper is available in the
``metadata``. Just like units can be multiplied with "raw" numbers/arrays to create
arrays with units, the helper can be multiplied or divided in to attach the "cosmo"
attributes.

.. code-block:: python

   dat = load("snap.hdf5")
   a = dat.metadata.a  # best give it a short name for ease of use
   densities = [1, 2, 3] * u.g / u.cm**3 * a.physical**-3

This specifies that these densities are in physical (not comoving) units and scale as
:math:`a^{-3}`. The shorter ``a.phys`` and ``a.com`` are also available as aliases for
``a.physical`` and ``a.comoving``, or for the especially keystroke-averse, ``a.p`` and
``a.c`` are also available.

.. warning::

   In older versions of :mod:`swiftsimio` the ``metadata.a`` attribute was an alias for
   ``metadata.scale_factor``, and stored a ``float``. Now, trying to use this helper as
   previously will raise an error, for example ``10 * metadata.a`` complains that you
   need to use ``metadata.a.physical`` or ``metadata.a.comoving``, not just
   ``metadata.a``. You can substitute ``metadata.scale_factor`` for a drop-in replacement.
   The one exception to this is that ``cosmo_array(..., scale_factor=metadata.a)`` will
   still work, to offer some backwards-compatibility.

You can also easily define cosmology-aware units with this helper, for example:

.. code-block:: python

   cMpc = u.Mpc * a.comoving  # equivalently: a.comoving**1, the power of 1 is implicit
   pMpc = u.Mpc * a.physical
   distance = 10 * pMpc
   number_density = 0.5 / cMpc**3

There is also a very similar :class:`~swiftsimio.objects.cosmo_quantity` class designed
for scalar values, analogous to the :class:`~unyt.array.unyt_quantity`. You may encounter
this being returned by :mod:`numpy` functions. Cosmology-aware scalar values can be
initialized similarly:

.. code-block:: python

   import unyt as u
   from swiftsimio.objects import cosmo_quantity, cosmo_factor

   my_cosmo_quantity = cosmo_quantity(
       2,
       u.Mpc,
       comoving=False,
       scale_factor=0.5,
       scale_exponent=1,
   )
   # or, equivalently:
   my_cosmo_quantity = 2 * u.Mpc * a.physical

