Halo Catalogues & SOAP integration
==================================

SWIFT-compatible halo catalogues, such as those written with SOAP, can be
loaded entirely transparently with ``swiftsimio``. It is generally possible
to use all of the functionality (masking, visualisation, etc.) that is used
with snapshots with these files, assuming the files conform to the
correct metadata standard.

An example SOAP file is available at
``http://virgodb.cosma.dur.ac.uk/swift-webstorage/IOExamples/soap_example.hdf5``

You can load SOAP files as follows:

.. code-block:: python

   from swiftsimio import load

   catalogue = load("soap_example.hdf5")

   print(catalogue.spherical_overdensity_200_mean.total_mass)

   # >>> [  591.      328.5     361.      553.      530.      507.      795.
   #        574.      489.5     233.75      0.     1406.      367.5    2308.
   #        ...
   #        0.      534.        0.      191.75   1450.      600.      290.   ] 10000000000.0*Msun (Physical)

What's going on here? Under the hood, ``swiftsimio`` has a discrimination function
between different metadata types, based upon a property stored in the HDF5 file,
``Header/OutputType``. If this is set to ``FullVolume``, we have a snapshot,
and use the :obj:`swiftsimio.metadata.objects.SWIFTSnapshotMetadata`
class. If it is ``SOAP``, we use
:obj:`swiftsimio.metadata.objects.SWIFTSOAPMetadata`, which instructs
``swiftsimio`` to read slightly different properties from the HDF5 file.

swiftgalaxy
-----------

The :mod:`swiftgalaxy` companion package to :mod:`swiftsimio` offers further integration with halo catalogues in SOAP, Caesar and Velociraptor formats (so far). It greatly simplifies efficient loading of particles belonging to an object from a catalogue, and additional tools that are useful when working with a galaxy or other localized collection of particles. Refer to the `swiftgalaxy documentation`_ for details.

.. _swiftgalaxy documentation: https://swiftgalaxy.readthedocs.io
