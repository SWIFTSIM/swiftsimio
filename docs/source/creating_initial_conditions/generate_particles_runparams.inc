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

