Power Spectra
=============

Though SWIFT includes functionality for calculating power spectra, this tool
runs on-the-fly, and as such after a run has completed you may wish to create
a number of more non-standard power spectra.

These tools are available as part of the :mod:`swiftsimio.visualisation.power_spectrum`
package. Making a power spectrum consists of two major steps: depositing the particles
on grid(s), and then binning their fourier transform to get the one-dimensional power.


Depositing on a Grid
--------------------

Depositing your particles on a grid is performed using
:meth:`swiftsimio.visualisation.power_spectrum.render_to_deposit`. This function
performs a nearest-grid-point (NGP) of all particles in the provided particle
dataset. For example:

.. code-block:: python

    from swiftsimio import load
    from swiftsimio.visualisation.power_spectrum import render_to_deposit

    data = load("cosmo_volume_example.hdf5")

    gas_mass_deposit = render_to_deposit(
        data.gas,
        resolution=512,
        project="masses",
        parallel=True,
    )

The specific field being depositied can be controlled with the ``project``
keyword argument. The ``resolution``` argument gives the one-dimensional
resolution of the 3D grid, so in this case you would recieve a ``512x512x512``
grid. Note that the ``gas_mass_deposit`` is a :obj:`swiftsimio.cosmo_array`,
and as such includes cosmological and unit information that is used later
in the process.


Generating a Power Spectrum
---------------------------

Once you have your grid deposited, you can easily generate a power spectrum
using the
:meth:`swiftsimio.visualisation.power_spectrum.deposition_to_power_spectrum`
function. For example, using the above deposit:

.. code-block:: python

    from swiftsimio.visualisation.power_spectrum import deposition_to_power_spectrum

    wavenumbers, power_spectrum, _ = deposition_to_power_spectrum(
        deposition=gas_mass_deposit,
        box_size=data.metadata.box_size,
    )

This power spectrum can then be plotted. Units are included on both the wavenumbers
and the power spectrum. Cross-spectra are also supported through the
``cross_deposition`` keyword, but by default this generates the auto power.
Wavenumbers are calculated to be at the weighted mean of the k-values in each
bin, rather than representing the center of the bin.


More Complex Scenarios
----------------------

In a realistic simualted power spectrum, you will need to perform 'folding'
to achieve a viable dynamic range within an achievable memory footprint.
Consider, for instance, a 1 Gpc simulation volume with a 1 kpc resolution
limit. In that case, you would need a deposition grid with :math:`10^{18}`
cells, amounting to a 4 EB (yes, exabyte!) memory footprint. That is,
of course, not realistic!

As in a power spectrum we are only interested in the periodicity of the
system, we can fold it back in on itself during the rendering process.
The position of the particle in the box is set to be:

:math:`x_i' := \left( x_i \% \frac{L}{2^{n}} \right) \frac{2^{n}}{L}`

where :math:`L` is the box-size and :math:`n` is some integer greater
than or equal to zero. This allows you to probe modes in the reduced
box-length :math:`L / 2^{n}` with the same fixed resolution deposition
buffer.

The ``folding`` parameter is available for both ``render_to_deposit``
and ``deposition_to_power_spectrum``, but it may be easier to use the
utility functions provided for automatically stitching together
the folded spectra. The function
:meth:`swiftsimio.visualsation.power_spectrum.folded_depositions_to_power_spectrum`
allows you to do this easily:

.. code-block:: python

    from swiftsimio.visualisation.power_spectrum import folded_depositions_to_power_spectrum
    import unyt

    folded_depositions = {}

    for folding in [x * 2 for x in range(5)]:
        folded_depositions[folding] = render_to_deposit(
            data.gas,
            resolution=512,
            project="masses",
            parallel=True,
            folding=folding,
        )

    bins, centers, power_spectrum, foldings = folded_depositions_to_power_spectrum(
        depositions=folded_depositions,
        box_size=data.metadata.box_size,
        number_of_wavenumber_bins=128,
        wavenumber_range=[1e-2 / unyt.Mpc, 1e2 / unyt.Mpc],
        log_wavenumber_bins=True,
        workers=4,
        minimal_sample_modes=8192,
        cutoff_above_wavenumber_fraction=0.75,
        shot_noise_norm=len(gas_mass_deposit),
        
    )

The 'used' foldings of the power spectrum are shown in the
``foldings`` return vaule, which is an array containing the folding
that was used for each given bin. This is useful for debugging and
visualisation.

There are a few crucial parameters to this function:

1. ``workers`` is the number of threads to use for the calculation of
   the fourier transforms.
2. ``minimal_sample_modes`` is the minimum number of modes that must be
    present in a bin for it to be included in the final power spectrum.
    Generally for a big simulation you want to set this to around 10'000,
    and this number is ignored for the lowest wavenumber bin.
3. ``cutoff_above_wavenumber_fraction`` is the fraction of the
   individual fold's (as represented by the FFT itself) maximally sampled
   wavenumber. Ignored for the last fold, and we always cap the maximal
   wavenumber to the nyquist frequency.
4. ``shot_noise_norm`` is the number of particles in the simulation
    that contribute to the power spectrum. This is used to normalise
    the power spectrum to the shot noise level. This is very
    important in this case because of the use of NGP deposition.
   
Foldings are stitched using a simple method where the 'better sampled'
foldings are used preferentially, up to the cutoff value.