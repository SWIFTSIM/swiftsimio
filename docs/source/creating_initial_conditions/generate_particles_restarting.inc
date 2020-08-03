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

