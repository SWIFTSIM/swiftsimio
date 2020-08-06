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
    generator.run_params.max_iterations = 2000
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
    w.gas.coordinates = generator.coordinates
    w.gas.masses = generator.masses
    w.gas.smoothing_length = generator.smoothing_lengths
    w.gas.densities = generator.densities

    # required to write IC files, but not generated:
    w.gas.velocities = np.random.random(number_of_particles**ndim) * unyt.cm / unyt.s
    w.gas.internal_energy = np.random.random(number_of_particles**ndim) * unyt.cm**2/unyt.s**2

    w.write("my_ic_file.hdf5")


This example finishes after 529 iterations and gives the following result:

.. image:: initial_conditions_example.png

