"""
Tests the initial condition generation.
"""

import pytest
import unyt
import numpy as np
from swiftsimio.initial_conditions import ParticleGenerator


class ICTemplate(object):
    """
    A class that holds templates of correct types of parameters.
    """

    def rho(self, x, ndim):
        return np.ones(x.shape[0])

    boxsize = np.ones(3) * unyt.cm
    unit_system = unyt.unit_systems.UnitSystem("name", "cm", "g", "s")
    nx = 10
    ndim = 2
    periodic = True
    eta = 1.2348
    rho_max = 1.0
    kernel = "cubic spline"

    def __init__(self):
        return


def test_initial_conditions_raising_exceptions():
    """
    Test that the proper exceptions are being raised
    """

    t = ICTemplate()
    g = ParticleGenerator(t.rho, t.boxsize, t.unit_system, t.nx, t.ndim,)

    with pytest.raises(TypeError):
        # check boxsize
        for boxsize in [
            "a",
            1,
            1.2,
            np.zeros(3),
            np.zeros(1),
            unyt.km,
            unyt.unit_systems.UnitSystem("x", "km", "g", "s"),
        ]:
            ParticleGenerator(
                t.rho, boxsize, t.unit_system, t.nx, t.ndim,
            )
        # check unit system
        for usys in [
            "a",
            1,
            1.2,
            np.zeros(3),
            np.zeros(1),
            unyt.km,
            np.ones(10) * unyt.s,
        ]:
            ParticleGenerator(
                t.rho, t.boxsize, usys, t.nx, t.ndim,
            )
        for n in ["a", 1.2, np.ones(1), None]:
            # check number of particles
            ParticleGenerator(
                t.rho, t.boxsize, t.unit_system, n, t.ndim,
            )
            # check number of dimensions
            ParticleGenerator(
                t.rho, t.boxsize, t.unit_system, t.nx, n,
            )

        def rho1(x):
            x[:, 0] += 1
            return x

        def rho2(x):
            return list(range(20))

        for rho in [rho1, rho2]:
            ParticleGenerator(rho, t.boxsize, t.unit_system, t.nx, t.ndim, t.rho_max)

        # test initial_setup routine
        for x in [
            "a",
            1,
            1.2,
            np.zeros(3),
            np.zeros(1),
            unyt.km,
            unyt.unit_systems.UnitSystem("x", "km", "g", "s"),
        ]:
            g.initial_setup(x=x)
        for m in [
            "a",
            1,
            1.2,
            np.zeros(3),
            np.zeros(1),
            unyt.km,
            unyt.unit_systems.UnitSystem("x", "km", "g", "s"),
        ]:
            g.initial_setup(m=m)

    with pytest.raises(ValueError):
        g.run_params.max_iterations = 10
        g.run_params.min_iterations = 11
        g.initial_setup()
        g.run_params.delta_init = 1e-5
        g.run_params.min_delta_r_norm = 1e-4
        g.initial_setup()

        def negative_rho(x, ndim):
            return -np.ones(x.shape[0])

        g.density_function = negative_rho
        g.initial_setup()


def test_initial_conditions_coordinate_generation():
    """
    Test initial coordinate generation
    """
    t = ICTemplate()
    g = ParticleGenerator(t.rho, t.boxsize, t.unit_system, t.nx, t.ndim,)

    # check zeroes in unused dimensions
    for ndim in [1, 2]:
        g.ndim = ndim
        x = g.generate_uniform_coords()
        assert np.all(x[:, ndim:] == 0)
        x = g.generate_displaced_uniform_coords()
        assert np.all(x[:, ndim:] == 0)
        x = g.rejection_sample_coords()
        assert np.all(x[:, ndim:] == 0)

    return


def test_initial_conditions_iteration_runs():
    """
    Run the iterations
    """

    t = ICTemplate()
    g = ParticleGenerator(t.rho, t.boxsize, t.unit_system, t.nx, t.ndim,)
    g.run_params.max_iterations = 2

    # run default parameters
    g.initial_setup()
    g.run_iteration()
    assert g.stats.number_of_iterations == g.run_params.max_iterations - 1
    # -1: number_of_iterations stores index in array, i.e. starts at 0

    # set min iterations
    g.run_params.max_iterations = 10
    g.run_params.min_iterations = 4
    # force iteration stop after 1 iteration
    g.run_params.displacement_threshold = 1e20
    g.run_params.unconverged_particle_number_tolerance = 1e20
    g.initial_setup()
    g.run_iteration()
    assert g.stats.number_of_iterations == g.run_params.min_iterations

    def rho_nonuniform(x, ndim):
        """
        use non-uniform density for futher testing
        forcing the algorithm to continue for 
        uniform initial setups can lead to bad things
        """
        return 1.0 - (x[:, 0] - 0.5) ** 2

    # try non-periodic run for every coord generation method
    # also check that all relevant parameters have been reset
    for method in ["uniform", "displaced", "rejection"]:
        for ndim in [1, 2, 3]:
            for periodic in [True, False]:
                g = ParticleGenerator(
                    rho_nonuniform,
                    t.boxsize,
                    t.unit_system,
                    int(100.0 ** (1.0 / ndim))+1,
                    ndim,
                    periodic=periodic,
                    eta=2,
                )
                g.run_params.min_iterations = 4
                g.run_params.max_iterations = 5
                g.run_params.particle_redistribution_frequency = 1
                g.run_params.particle_redistribution_number_fraction = 0.2
                g.run_params.state_dump_frequency = 4  # do one dump
                g.run_params.state_dump_basename = "restart"

                g.initial_setup(method=method, x=None, m=None)
                g.run_iteration()

                # now check restarts
                r = ParticleGenerator(
                    rho_nonuniform,  # this mustn't be changed
                    t.boxsize * 2,
                    unyt.unit_systems.UnitSystem(":(", "pc", "kg", "hs"),
                    3,
                    ndim + 1,
                )
                r.restart("restart_00004.hdf5")
                g.run_params.min_iterations = 1
                g.run_params.max_iterations = 1

                assert r.npart == g.npart
                assert r.ndim == g.ndim
                for d in range(ndim):
                    assert r.boxsize[d] == g.boxsize[d]
                assert r.eta == g.eta
                assert r.periodic == g.periodic
