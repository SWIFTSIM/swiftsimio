Py-SPHViewer Integration
========================

We provide a wrapper of the ever-popular py-sphviewer_ for easy use with
:mod:`swiftsimio` datasets. Particle datasets that do not contain smoothing
lengths will have them generated through the use of the scipy ``cKDTree``.
You can get access to the objects through a sub-module as follows:

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.sphviewer import SPHViewerWrapper

   data = load("my_snapshot_0000.hdf5")

   resolution = 2048

   gas = SPHViewer(data.gas, smooth_over="masses")
   gas_temp = SPHViewer(
       data.gas,
       smooth_over=data.gas.masses * data.gas.temperatures
   )
   dark_matter = SPHViewer(data.dark_matter, smooth_over="masses")

   gas.quick_view(xsize=resolution, ysize=resolution, r="infinity")
   gas_temp.quick_view(xsize=resolution, ysize=resolution, r="infinity")
   dark_matter.quick_view(xsize=resolution, ysize=resolution, r="infinity")

   plt.imsave("gas_image.png", gas.image)
   plt.imsave("gas_temp.png", gas_temp.image / gas.image)
   plt.imsave("dm_image.png", dark_matter.image)


The :obj:`swiftsimio.visualisation.sphviewer.SPHViewerWrapper` object allows you
to get access to the particles, camera, and render object through ``.particles``,
``.get_camera()`` and ``.camera``, and ``.get_render()`` and ``.render``
respectively.

.. _py-sphviewer: https://github.com/alejandrobll/py-sphviewer