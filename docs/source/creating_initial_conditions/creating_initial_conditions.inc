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

Secondly, what constant :math:`C_{\Delta r}` should we use in eq. 
:eq:`delta_r`?  In the code, it is defined in units of the mean interparticle 
distance

.. math::

    \overline{l} = 1 / N^{1/\nu}



How large the :math:`\Delta r` without the constant :math:`C_{\Delta r}` will be 
depends on multiple factors, like what density function you're trying to 
reproduce, how many neighbours you include in your kernel summation, etc.
You should set it in a way such that the displacements at the start of the 
iteration are no larger than the order of unity in units of :math:`\overline{l}`.






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







Additional and Runtime Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The methods used in the example above accept a handful of optional arguments.
The most commonly used ones are:

.. code-block:: python

    from swiftsimio.initial_conditions import ParticleGenerator

    # we assume the required arguments are set as in the example above
    generator = ParticleGenerator(
        rho,
        boxsize,
        unit_system,
        number_of_particles,
        ndim,
        periodic = True, # whether the simulation is periodic
        kernel = 'cubic spline' # which kernel to use
        eta = 1.2348 # resolution eta; determines number of neighbours used
    )


    generator.initial_setup(
        method = 'rejection', # what method to use for initial particle coordinates guess
                              # options: 'rejection', 'uniform', 'displaced'
        x = None,   # explicitly give which initial particle configuration to use
        m = None,   # explicitly give which particle masses to use
    )



For more details, look into the API Documentation of :py:class:`~.initial_conditions.generate_particles.ParticleGenerator`
and :py:func:`~swiftsimio.initial_conditions.generate_particles.ParticleGenerator.initial_setup()`


Furthermore, quite a few parameters concerning the iterations can be set manually:

.. code-block:: python

    from swiftsimio.initial_conditions import ParticleGenerator

    # we assume the required arguments are set as in the example above
    generator = ParticleGenerator(
        rho,
        boxsize,
        unit_system,
        number_of_particles,
        ndim,
    )


    # stop after this many iterations
    generator.run_params.max_iterations = 1000
    # don't stop before this many iterations
    generator.run_params.min_iterations = 0
    # how far the particles are allowed to move to be considered 
    # converged. In units of mean interparticle distance
    generator.run_params.convergence_threshold = 1e-4
    # fraction of particle allowed to move at a higher distance than
    # convergence_threshold and still be considered converged
    generator.run_params.unconverged_particle_number_tolerance = 5e-3
    # you haven't converged while there are particles that are
    # displaced with this distance, in units of mean interp. distance
    generator.run_params.displacement_threshold = 1e-3
    # initial normalisation constant for displacement force in units of
    # mean interparticle distance. If None, it will be set such that
    # the maximal displacement in the first iteration is set to 1 mean
    # interparticle distance.
    generator.run_params.delta_init = None
    # multiply the displacement force normalisation with this factor 
    # after each iteration
    generator.run_params.delta_r_norm_reduction_factor = 0.99
    # don't go below this normalisation, in units of mean interp. distance
    generator.run_params.min_delta_r_norm = 1e-6
    # redistribute particles after this many iterations
    generator.run_params.particle_redistribution_frequency = 40
    # what fraction of total particles to aim to redistribute
    generator.run_params.particle_redistribution_number_fraction = 0.01
    # reduce the fraction of particles redistributed after
    # every redistribution
    generator.run_params.particle_redistribution_number_reduction_factor = 1.
    # don't redistribute particles after this many iterations
    generator.run_params.no_particle_redistribution_after = 200
    # dump intermediate states of the IC generations. Can be used for
    # checks of progress and for restarts.
    generator.run_params.state_dump_frequency = 50
    # basename for intermadiate state dump files
    generator.run_params.state_dump_basename = "IC_generation_iteration"
    # set a specific random seed
    generator.run_params.set_random_seed(20)


    # make sure to call initial_setup *after* you change run params!
    generator.initial_setup()

    # now you do your thing.
    generator.run_iteration()
    ...


For more details, look into the :py:class:`~.initial_conditions.generate_particles.RunParams` 
class API documentation.








Restarting
~~~~~~~~~~

Sometimes you'd like the generation of initial conditions to run for more steps
than initially set up, or for example stop the reduction of the normalisation
constant earlier than initially anticipated. For such occiasions, the iteration
can be restarted from some advanced stage.

There are two ways of doing this. The better way of doing this is to use the
:py:meth:`~.initial_conditions.generate_particles.ParticleGenerator.restart`
method. For it to work, you need to provide it with an intermediate state dump
file that the code creates. To have the code generate them, you need to set
``ParticleGenerator.run_params.state_dump_frequency`` to be greater than zero:

.. code-block:: python

    from swiftsimio.initial_conditions import ParticleGenerator
    
    # set up generator
    generator = ParticleGenerator(...)

    # make sure you get intermediate state dumps
    generator.run_params.state_dump_frequency = 1
    # set up dump file basename
    generator.run_params.state_dump_basename = "ic_dump"
    # stop e.g. at iteration 2 for demonstration purposes
    generator.run_params.max_iterations = 2

    # set up and run
    generator.initial_setup()
    generator.run_iteration()

    # this runs for 2 iterations and dumps the file 'ic_dump_00002.hdf5'


    # now let's restart!

    # you'll have to create a new particle generator instance.
    # All the given required parameters will be overwritten by the
    # restart operation, except for the function rho. This one
    # you'll need to provide!
    restart_generator = ParticleGenerator(...)

    restart_generator.restart("ic_dump_00002.hdf5")
    restart_generator.run_iteration()



After calling ``restart_generator.restart()``, you may still tweak run
parameters like ``restart_generator.run_params.whatever`` as you wish. In fact,
that might be necessary if the generation ended because your previously set
convergence criteria have been met. However, note that the restart operation is 
set up such that the normalisation constant is the same as in the previous run. 
If you change or even explicitly set ``restart_generator.run_params.delta_init``, 
that information will be lost.

**Note**: The iteration count will restart at zero again. So be careful not to
overwrite data that you still might want to keep!

If you don't want to dump too many intermediate states, but still would like to
retain the possibility to restart at the end, it is recommended to set
``generator.run_params.state_dump_frequency`` to some ridiculously high integer.
As long as it is > 0, it will create a single dump at the end of the run, 
precisely for restarting purposes.


If you opted out of creating intermediate state dumps, another way of restarting
is by reading in the initial condition file that has been written in the first
run and pass on the coordinates and particle masses to the
:py:meth:`~.initial_conditions.generate_particles.ParticleGenerator.initial_setup`
function:


.. code-block: python

    from swiftsimio.initial_conditions import ParticleGenerator

    # make sure to feed in the same data as in the first run!
    generator = ParticleGenerator(...)

    # set up run parameters as you want
    generator.run_params.whatever = whatever_else

    from swiftsimio import load
    data = load("my_ic_file.hdf5")
    x = data.gas.coords
    m = data.gas.masses

    generator.initial_setup(x=x, m=m)

    # and off we go!
    generator.run_iteration()










Full Example
~~~~~~~~~~~~


.. code-block:: python

    import numpy as np
    import unyt
    from swiftsimio.initial_conditions import ParticleGenerator
    from swiftsimio import Writer

    def rho(x, ndim):
        """
        A sine wave.
        """
        return 0.2 * (1.05 + np.sin(2 * np.pi * x[:, 0]))

    number_of_particles = 100 # number of particles along every dimension
    ndim = 2 # number of dimensions for the initial conditions
    unit_system = unyt.unit_systems.UnitSystem("I name thee Sir Theodore", 'cm', 'g', 's') # a unit system
    boxsize = unyt.unyt_array([1., 1., 1.], "cm") # a box size

    generator = ParticleGenerator(
                rho,
                boxsize,
                unit_system,
                number_of_particles,
                ndim,
            )

    # set up run parameters
    generator.run_params.max_iterations = 1
    generator.run_params.convergence_threshold = 1e-4
    generator.run_params.unconverged_particle_number_tolerance = 5e-3
    generator.run_params.displacement_threshold = 1e-3
    generator.run_params.delta_init = None
    generator.run_params.delta_r_norm_reduction_factor = 0.99
    generator.run_params.min_delta_r_norm = 1e-6
    generator.run_params.particle_redistribution_frequency = 40
    generator.run_params.particle_redistribution_number_fraction = 0.01
    generator.run_params.particle_redistribution_number_reduction_factor = 1.
    generator.run_params.no_particle_redistribution_after = 200
    generator.run_params.state_dump_frequency = 50
    generator.run_params.set_random_seed(20)

    # run some internal setups
    generator.initial_setup()

    # run the iterations
    generator.run_iteration()


    # Finally, write down the data
    w = Writer(unit_system, boxsize)

    # accessing the results from the generator
    w.gas.coordinates = generator.coords
    w.gas.masses = generator.masses
    w.gas.smoothing_length = generator.smoothing_lengths
    w.gas.densities = generator.densities

    # required to write IC files, but not generated:
    w.gas.velocities = np.random.random(number_of_particles**ndim) * unyt.cm / unyt.s
    w.gas.internal_energy = np.random.random(number_of_particles**ndim) * unyt.cm**2/unyt.s**2

    w.write("my_ic_file.hdf5")


This example finishes after 529 iterations and gives the following result:

.. image:: initial_conditions_example.png
