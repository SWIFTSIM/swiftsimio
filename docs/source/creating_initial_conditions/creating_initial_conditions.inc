Creating Hydro Initial Conditions
---------------------------------

An elegant way of generating hydrodynamics initial conditions for SPH 
simulations is described by `Arth et al 2019 <https://arxiv.org/abs/1907.11250>`_.
The essential idea is to start with some initial particle configuration, then 
compute some sort of 'force' on the particles that will displace them towards 
the model density, and re-iterate until the configuration has relaxed 
sufficiently.

Before the iteration can begin, two things need to be taken care of. Firstly, 
the particle masses need to be determined. It is in our interest that the SPH 
particles have equal masses, hence we require the total mass in the simulation 
box as determined by the given density. To this end, the density field is 
integrated numerically, and the total mass divided equally amongst the 
particles.
Secondly, some initial guess for the particle coordinates needs to be created
so that the algorithm can start off somewhere. By default, this is done by
rejection sampling the model density function, but other options are available
(see :py:meth:`~.swiftsimio.initial_conditions.generate_particles.ParticleGenerator.initial_setup`)
Finally, if there is some other particle coordinates or masses that you'd prefer
to use, particle coordinates and masses can be provided directly to the 
:py:meth:`~.swiftsimio.initial_conditions.generate_particles.ParticleGenerator.initial_setup`
method.



Every iteration step follows these steps:

-   Find the neighbours of each particle, and compute a 'displacement force' 
    on each particle. The 'displacement force' experienced by particle :math:`i` 
    due to particle :math:`j` is given by
        
.. math::
    :label: delta_r

    \Delta r = C_{\Delta r} h_{ij} W(|\mathbf{x}_i - \mathbf{x}_j|, h_{ij}) 
    \frac{\mathbf{x}_i - \mathbf{x}_j}{|\mathbf{x}_i - \mathbf{x}_j|} 

where :math:`C_{\Delta r}` is a constant discussed later, and 
        
.. math::
 
    h_{ij} = \frac{h_i + h_j}{2}
        

and only contributions from all neighbours :math:`j` of particle :math:`i` are 
taken into account. :math:`h_{i}` are not actual smoothing lengths as they
would be used in a SPH simulation, but model smoothing lengths based on the
model density :math:`\rho_m` that is to be simulated.
        
-   Move particles with the computed :math:`\Delta r`
    
-   Optionally, displace some overdense particles close to underdense particles. 
    Typically, this is not done every step, but after some set number of iteration 
    steps have passed. A user-set fraction of overdense particles is selected to be 
    moved, based on a random choice weighted by the ratio of the current particle 
    density to the model density at that position such that more overdense particles
    are more likely to be moved to more underdense particles' viscinities.
    Once a target for some overdense particle is decided, the overdense particle is 
    placed randomly around the target's coordinates with distance :math:`< 0.3` the 
    kernel support radius.

When the initial conditions start to converge, the forces :math:`\Delta r` 
decrease. The first condition for convergence is that an upper threshold for 
any displacement is never reached. If that is satisfied, we may consider the 
initial conditions to be converged when a large fraction (e.g. 99% or 
99.9% or...) of the particles has a displacement lower than some 
convergence displacement threshold, which typically should be lower than the 
upper threshold for any displacement. Finally, the iteration may stop if some 
maximal number of iterations has been completed.

The normalisation constant :math:`C_{\Delta r}` in eq. :eq:`delta_r` is 
defined in units of the mean interparticle distance in the code.

How large the :math:`\Delta r` without the constant :math:`C_{\Delta r}` will be 
depends on multiple factors, like what density function you're trying to 
reproduce, how many neighbours you include in your kernel summation, etc.
You should set it in a way such that the displacements at the start of the 
iteration are no larger than the order of unity of the mean interparticle
distance. If you don't specify it, it will automatically be set up such that the
maximal displacement of the first iteration will be exactly one mean
interparticle distance.






User's Guide
^^^^^^^^^^^^

Required Parameters
~~~~~~~~~~~~~~~~~~~

The functionality to create initial condition is available through the
:py:mod:`swiftsimio.initial_conditions` submodule, and the top-level
:py:class:`swiftsimio.initial_conditions.generate_particles.ParticleGenerator` object.

There are five required arguments that must be provided:

- ``rho``: The model density that is to be generated. It must be a function
  that takes exactly two arguments: A ``numpy.ndarray x`` with shape ``(npart, 3)``
  regardless of how many dimensions are to be used for the simulation, and 
  ``int ndim``, the number of dimensions to be used. It also must return a
  ``numpy.ndarray`` with the shape ``(npart)`` of the model densities based
  on the provided particle positions ``x``.
- ``boxsize``: A ``unyt_array`` that contains the boxsize to be used.
- ``unit_system`` : A ``unyt.unit_systems.UnitSystem`` object that contains
  the units that the coordinates, masses, and densities should be computed in.
- ``number_of_particles``: The number of particles to be used in every dimension.
  In total, there will be ``number_of_particles ^ ndim`` particles used in the
  initial conditions.
- ``ndim``: The number of dimensions to be used for the initial conditions.



**Example**:


.. code-block:: python

    import numpy as np
    import unyt

    # the model density to be generated
    def rho(x, ndim):
        """A sine wave in x direction."""
        return 1.1 + np.sin(2 * np.pi * x[:, 0])

    boxsize = unyt.unyt_array([1.0, 1.0, 1.0], "cm")  # a box size
    unit_system = unyt.unit_systems.UnitSystem("name", "cm", "g", "s")  # a unit system
    number_of_particles = 100  # number of particles along every dimension
    ndim = 2  # number of dimensions for the initial conditions






Basic Workflow
~~~~~~~~~~~~~~

The essential steps are as follows:

.. code-block:: python

    from swiftsimio.initial_conditions import ParticleGenerator

    # set up the particle generator
    # we assume the required arguments are set as in the example above
    generator = ParticleGenerator(
        rho,
        boxsize,
        unit_system,
        number_of_particles,
        ndim,
    )

    # run some internal setups
    generator.initial_setup()

    # run the iterations
    generator.run_iteration()


    # Finally, write down the data
    from swiftsimio import Writer
    w = Writer(unit_system, boxsize)

    # accessing the results from the generator
    w.gas.coordinates = generator.coords
    w.gas.masses = generator.masses
    w.gas.smoothing_length = generator.smoothing_lengths
    w.gas.densities = generator.densities

    # required to write IC files, but not generated:
    w.gas.velocities = ... # put in here whatever you need;
    w.gas.internal_energy = ... # put in here whatever you need;

    w.write("my_ic_file.hdf5")
