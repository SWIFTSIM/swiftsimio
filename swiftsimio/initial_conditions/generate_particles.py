"""Generate SPH initial conditions for SPH simulations iteratively for a given density function following Arth et al.
2019 (https://arxiv.org/abs/1907.11250).

"""

# --------------------------------------------
# IC generation related stuff
# --------------------------------------------


import numpy as np
import unyt
from unyt import unyt_array
from math import erf
from typing import Union

from .IC_kernel import get_kernel_data
from swiftsimio.optional_packages import KDTree, TREE_AVAILABLE
from swiftsimio import Writer


seed_set = False


class ic_sim_params(object):
    r"""
    object containing simulation parameters
    """
    boxsize: unyt_array = unyt_array([1.0, 1.0, 1.0], "cm")
    periodic: bool = True
    nx: int = 100
    ndim: int = 2
    unit_l: Union[None, unyt.unit_object.Unit, str] = None
    unit_m: Union[unyt.unit_object.Unit, str] = "g"
    kernel: str = "cubic spline"
    eta: float = 1.2348

    def __init__(
        self,
        boxsize: unyt_array = unyt_array([1.0, 1.0, 1.0], "cm"),
        periodic: bool = True,
        nx: int = 100,
        ndim: int = 2,
        unit_l: Union[None, unyt.unit_object.Unit, str] = None,
        unit_m: Union[unyt.unit_object.Unit, str] = "g",
        kernel: str = "cubic spline",
        eta: float = 1.2348,
    ):
        r"""
        Set up the simulation parameters for the initial conditions you want to
        generate. The dict it returns is a necessary argument to call
        ``generate_IC_for_given_density()``


        Parameters
        -------------

        boxsize: unyt_array, optional
            The box size of the simulation.

        periodic: bool, optional
            whether the simulation box is periodic or not

        nx: int, optional
            how many particles along every dimension you want your simulation
            to contain

        ndim: int, optional
            how many dimensions you want your simulation to have

        unit_l:  None or unyt.unit_object.Unit or str, optional
            unit length for the coordinates. If None, the same unit given for
            the boxsize will be used.

        unit_m: unyt.unit_object.Unit or str, optional
            unit mass for the particle masses

        kernel: str {'cubic spline',}, optional
            which kernel to use

        eta: float, optional
            resolution eta, which defines the number of neighbours used
            independently of dimensionality


        Returns
        ------------

        ic_sim_params: dict
            dict containing the parameters stored in a way such that ``generate_IC_for_given_density()``
            understands them.


        Notes
        -----------
        
        + The returned dict is a required argument to call ``generate_IC_for_given_density()``
        """

        self.boxsize = boxsize
        self.periodic = periodic
        self.nx = nx
        self.ndim = ndim
        if unit_l is None:
            self.unit_l = boxsize.units
            self.boxsize_to_use = boxsize.value  # use only np.array
        else:
            self.unit_l = unit_l
            self.boxsize_to_use = boxsize.to(unit_l).value
        self.unit_m = unit_m
        self.kernel = kernel
        self.eta = eta
        return

    def copy(self):
        r"""
        Returns a copy of itself.
        """
        new = ic_sim_params(
            boxsize=self.boxsize,
            periodic=self.periodic,
            nx=self.nx,
            ndim=self.ndim,
            unit_l=self.unit_l,
            unit_m=self.unit_m,
            kernel=self.kernel,
            eta=self.eta,
        )
        return new


class ic_run_params(object):
    r"""
    Objects containing run parameters for IC generation
    """
    iter_max: int = 2000
    converge_thresh: float = 1e-4
    tolerance_part: float = 1e-3
    displ_thresh: float = 1e-3
    delta_init: Union[float, None] = None
    delta_reduct: float = 1.0
    delta_min: float = 1e-6
    redist_freq: int = 20
    redist_frac: float = 0.01
    redist_reduct: float = 1.0
    no_redistribution_after: int = 200
    dump_basename: str = "IC-generation-iteration-"
    dumpfreq: int = 50

    def __init__(
        self,
        iter_max: int = 2000,
        convergence_threshold: float = 1e-4,
        tolerance_part: float = 1e-3,
        displacement_threshold: float = 1e-3,
        delta_init: Union[float, None] = None,
        delta_reduction_factor: float = 1.0,
        delta_min: float = 1e-6,
        redistribute_frequency: int = 20,
        redistribute_fraction: float = 0.01,
        redistribute_fraction_reduction: float = 1.0,
        no_redistribution_after: int = 200,
        intermediate_dump_frequency: int = 50,
        dump_basename: str = "IC-generation-iteration-",
        random_seed: int = 666,
    ):
        r"""
        Set up the runtime parameters for the initial condition generation.
        The dict it returns is a necessary argument when calling 
        ``generate_IC_for_given_density()``.

        Parameters
        --------------

        iter_max: int, optional
            max numbers of iterations for generating IC conditions

        convergence_threshold: float, optional
            if enough particles are displaced by distance below threshold * mean 
            interparticle distance, stop iterating.  
        
        tolerance_part: float, optional
            tolerance for not converged particle fraction: this fraction of 
            particles can be displaced with distances > threshold

        displacement_threshold: float, optional      
            iteration halt criterion: Don't stop until every particle is displaced 
            by distance < threshold * mean interparticle distance

        delta_init: float or None, optional
            initial normalization constant for particle motion in units of mean 
            interparticle distance. If ``None`` (default), delta_init will be set 
            such that the maximal displacement found in the first iteration is 
            normalized to 1 mean interparticle distance.

        delta_reduction_factor: float, optional
            multiply the normalization constant for particle motion by this factor 
            after every iteration. In certain difficult cases this might help the 
            generation to converge if set to < 1.
        
        delta_min: float, optional
            minimal normalization constant for particle motion in units of mean 
            interparticle distance.
       
        redistribute_frequency: int, optional 
            redistribute a handful of particles every ``redistribute_frequency`` 
            iteration. How many particles are redistributed is controlled with the 
            ``redistribute_fraction`` parameter.

        redistribute_fraction: float, optional
            fraction of particles to be redistributed when doing so. 

        redistribute_fraction_reduction: float, optional
            multiply the ``redistribute_fraction`` parameter by this factor every 
            time particles are being redistributed. In certain difficult cases this
            might help the generation to converge if set to < 1.
        
        no_redistribution_after: int, optional     
            don't redistribute particles after this iteration.
     
        intermediate_dump_frequency: int, optional
            frequency of dumps of the current state of the iteration. If set to zero,
            no intermediate results will be stored.

        dump_basename: str
            Basename for intermediate dumps. The filename will be constructed as
            ``basename + <5 digit zero padded iteration number> + .hdf5``

        random_seed: int, optional
            set a specific random seed



        Returns
        ------------

        ic_run_params: dict
            dict containing the parameters stored in a way such that 
            ``generate_IC_for_given_density()`` understands them.


        Notes
        -----------
        
        + The returned dict is a required argument to call 
            ``generate_IC_for_given_density()``
        """

        self.iter_max = iter_max
        self.converge_thresh = convergence_threshold
        self.tolerance_part = tolerance_part
        self.displ_thresh = displacement_threshold
        self.delta_init = delta_init
        self.delta_reduct = delta_reduction_factor
        self.delta_min = delta_min
        self.redist_freq = redistribute_frequency
        self.redist_frac = redistribute_fraction
        self.redist_stop = no_redistribution_after
        self.redist_reduct = redistribute_fraction_reduction
        self.dumpfreq = intermediate_dump_frequency
        self.dump_basename = dump_basename

        global seed_set
        if not seed_set:
            seed_set = True
            np.random.seed(random_seed)

        return


def IC_uniform_coordinates(ic_sim_params: ic_sim_params):
    r"""
    Generate coordinates for a uniform particle distribution.


    Parameters
    ------------------

    ic_sim_params: swiftsimio.initional_conditions.ic_sim_params
        an ``ic_sim_params`` instance containing simulation parameters


    Returns
    
    ------------------

    x: unyt_array 
        unyt_array(shape=(npart, 3), dtype=float) of coordinates, where 
        ``npart = nx ** ndim``, both of which are set in the ``ic_sim_params``
        dict
    """

    nx = ic_sim_params.nx
    boxsize = ic_sim_params.boxsize_to_use
    ndim = ic_sim_params.ndim

    npart = nx ** ndim
    x = unyt_array(np.zeros((npart, 3), dtype=np.float), ic_sim_params.unit_l)

    dxhalf = 0.5 * boxsize[0] / nx
    dyhalf = 0.5 * boxsize[1] / nx
    dzhalf = 0.5 * boxsize[2] / nx

    if ndim == 1:
        x[:, 0] = np.linspace(dxhalf, boxsize[0] - dxhalf, nx)

    elif ndim == 2:
        xcoords = np.linspace(dxhalf, boxsize[0] - dxhalf, nx)
        ycoords = np.linspace(dyhalf, boxsize[1] - dyhalf, nx)
        for i in range(nx):
            start = i * nx
            stop = (i + 1) * nx
            x[start:stop, 0] = xcoords
            x[start:stop, 1] = ycoords[i]

    elif ndim == 3:
        xcoords = np.linspace(dxhalf, boxsize[0] - dxhalf, nx)
        ycoords = np.linspace(dyhalf, boxsize[1] - dyhalf, nx)
        zcoords = np.linspace(dzhalf, boxsize[2] - dzhalf, nx)
        for j in range(nx):
            for i in range(nx):
                start = j * nx ** 2 + i * nx
                stop = j * nx ** 2 + (i + 1) * nx
                x[start:stop, 0] = xcoords
                x[start:stop, 1] = ycoords[i]
                x[start:stop, 2] = zcoords[j]

    return x


def IC_perturbed_coordinates(ic_sim_params: ic_sim_params, max_displ: float = 0.4):
    """
    Get the coordinates for a randomly perturbed uniform particle distribution.
    The perturbation won't exceed ``max_displ`` times the interparticle distance
    along an axis.


    Parameters
    ------------------

    ic_sim_params: ic_sim_params object
        an ic_sim_params object instance containing simulation parameters

    max_displ: float
        maximal displacement of a particle initially on an uniform grid along 
        any axis, in units of particle distance along that axis


    Returns
    ------------------

    x: unyt_array 
        unyt_array(shape=(npart, 3), dtype=float) of coordinates, where 
        ``npart = nx ** ndim``, both of which are set in the ``ic_sim_params`` 
    """

    nx = ic_sim_params.nx
    boxsize = ic_sim_params.boxsize_to_use
    ndim = ic_sim_params.ndim
    periodic = ic_sim_params.periodic
    npart = nx ** ndim

    # get maximal deviation from uniform grid of any particle along an axis
    maxdelta = max_displ * boxsize / nx

    # generate uniform grid (including units) first
    x = IC_uniform_coordinates(ic_sim_params)

    for d in range(ndim):
        amplitude = unyt_array(
            np.random.uniform(low=-boxsize[d], high=boxsize[d], size=npart)
            * maxdelta[d],
            x.units,
        )
        x[:, d] += amplitude

        if periodic:  # correct where necessary
            over = x[:, d] > boxsize[d]
            x[over, d] -= boxsize[d]
            under = x[:, d] < 0.0
            x[under] += boxsize[d]
        else:
            # get new random numbers where necessary
            xmax = x[:, d].max()
            xmin = x[:, d].min()
            amplitude_redo = None
            while xmax > boxsize[d] or xmin < 0.0:

                over = x[:, d] > boxsize[d]
                under = x[:, d] < 0.0
                redo = np.logical_or(over, under)

                if amplitude_redo is None:
                    # for first iteration, get array in proper shape
                    amplitude_redo = amplitude[redo]

                # first restore previous state
                x[redo, d] -= amplitude_redo

                # then get new guesses, but only where necessary
                nredo = x[redo, d].shape[0]
                amplitude = unyt_array(
                    np.random.uniform(low=-boxsize[d], high=boxsize[d], size=nredo)
                    * maxdelta[d],
                    x.units,
                )
                x[redo, d] += amplitude_redo

                xmax = x[:, d].max()
                xmin = x[:, d].min()

    return x


def IC_sample_coordinates(
    ic_sim_params: ic_sim_params, rho_anal: callable, rho_max: Union[None, float] = None
):
    r"""
    Generate an initial guess for particle coordinates by rejection sampling the
    density


    Parameters
    ------------------

    ic_sim_params: ic_sim_params
        an ic_sim_params instance containing simulation parameters

    rho_anal: callable
        The density function that is to be reproduced in the initial conditions.
        It must take two positional arguments:
        
        - ``x``: np.ndarray of 3D particle coordinates (even if your initial 
          conditions have lower dimensionality)
        - ``ndim``: integer, number of dimensions that your simulations is to 
          have

    rho_max: float or None, optional
        The maximal density within the simulation box. If ``None``, an 
        approximate value will be determined.


    Returns
    ------------------

    x: unyt_array 
        unyt_array(shape=(npart, 3), dtype=float) of coordinates, where 
        ``npart = nx ** ndim``, both of which are set in the ``ic_sim_params`` 
        dict
    """

    nx = ic_sim_params.nx
    boxsize = ic_sim_params.boxsize_to_use
    ndim = ic_sim_params.ndim
    periodic = ic_sim_params.periodic
    npart = nx ** ndim

    x = np.empty((npart, 3), dtype=np.float)

    if rho_max is None:
        # find approximate peak value of rho_max
        nc = 200  # don't cause memory errors with too big of a grid. Also don't worry too much about accuracy.
        tempparams = ic_sim_params.copy()
        tempparams.nx = nc
        xc = IC_uniform_coordinates(tempparams)
        rho_max = rho_anal(xc.value, ndim).max() * 1.05
        # * 1.05: safety measure to make sure you're always above the analytical value

    keep = 0
    coord_threshold = boxsize
    while keep < npart:

        xr = np.zeros((1, 3), dtype=np.float)
        for d in range(ndim):
            xr[0, d] = np.random.uniform(low=0.0, high=coord_threshold[d])

        if np.random.uniform() <= rho_anal(xr, ndim) / rho_max:
            x[keep] = xr
            keep += 1

    return unyt_array(x, ic_sim_params.unit_l)


def generate_IC_for_given_density(
    rho_anal: callable,
    ic_sim_params: ic_sim_params,
    ic_run_params: ic_run_params,
    x: Union[unyt_array, None] = None,
    m: Union[unyt_array, None] = None,
    rho_max: Union[float, None] = None,
):
    """
    Generate SPH initial conditions for SPH simulations iteratively for a given
    density function ``rho_anal()`` following Arth et al. 2019 
    (https://arxiv.org/abs/1907.11250).


    Parameters
    ------------------

    rho_anal: callable
        The density function that is to be reproduced in the initial conditions.
        It must take two positional arguments:
        
        - ``x``: np.ndarray of 3D particle coordinates (even if your initial 
          conditions have lower dimensionality)
        - ``ndim``: integer, number of dimensions that your simulations is to 
          have

    ic_sim_params: ic_sim_params
        an ic_sim_params instance containing simulation parameters

    ic_run_params: ic_run_params
        an ic_run_params instance containing simulation parameters

    x: unyt_array or None, optional
        Initial guess for coordinates of particles. If ``None``, the initial 
        guess will be generated by rejection sampling the density function 
        ``rho_anal``

    m: unyt_array or None, optional
        ``unyt_array`` of particle masses. If ``None``, an array will be created
        such that the total mass in the simulation box given the analytical 
        density is reproduced, and all particles will have equal masses.

    rho_max: float or None, optional
        The maximal density within the simulation box. If ``None``, an 
        approximate value will be determined.
 

    Returns
    -----------

    coords: unyt_array
        Final particle coordinates

    masses: unyt_array
        Final particle masses

    stats: dict
        dict containing particle motion statistics of the last iteration:

        - ``stats['min_motion']``: The smallest displacement a particle 
          experienced in units of mean interparticle distance
        - ``stats['avg_motion']``: The average displacement particles 
          experienced in units of mean interparticle distance
        - ``stats['max_motion']``: The maximal displacement a particle 
          experienced in units of mean interparticle distance
        - ``stats['niter']``: Number of iterations performed
    """

    # safety checks first

    if not TREE_AVAILABLE:
        raise ImportError(
            "The scipy.spatial.cKDTree class is required to search for smoothing lengths."
        )

    try:
        res = rho_anal(np.ones((10, 3), dtype=np.float), ic_sim_params.ndim)
    except TypeError:
        errmsg = (
            "rho_anal must take only a numpy array x of coordinates as an argument."
        )
        raise TypeError(errmsg)
    if not isinstance(res, np.ndarray):
        raise TypeError("rho_anal needs to return a numpy array as the result.")

    # shortcuts and constants
    nx = ic_sim_params.nx
    ndim = ic_sim_params.ndim
    periodic = ic_sim_params.periodic
    boxsize = ic_sim_params.boxsize_to_use
    npart = nx ** ndim

    mid = 1.0
    for d in range(ndim):
        mid *= boxsize[d]
    mid = mid ** (1.0 / ndim) / nx  # mean interparticle distance

    if ic_run_params.delta_init is None:
        compute_delta_norm = True
        delta_r_norm = mid
    else:
        compute_delta_norm = False
        delta_r_norm = ic_run_params.delta_init * mid

    delta_r_norm_min = ic_run_params.delta_min * mid

    if periodic:  #  this sets up whether the tree build is periodic or not
        boxsize_for_tree = boxsize[:ndim]
    else:
        boxsize_for_tree = None

    # kernel data
    kernel_func, _, kernel_gamma = get_kernel_data(ic_sim_params.kernel, ndim)

    # expected number of neighbours
    if ndim == 1:
        Nngb = 2 * kernel_gamma * ic_sim_params.eta
    elif ndim == 2:
        Nngb = np.pi * (kernel_gamma * ic_sim_params.eta) ** 2
    elif ndim == 3:
        Nngb = 4 / 3 * np.pi * (kernel_gamma * ic_sim_params.eta) ** 3

    # round it up for cKDTree
    Nngb_int = int(Nngb + 0.5)

    # generate first positions if necessary
    if x is None:
        x = IC_sample_coordinates(ic_sim_params, rho_anal, rho_max)

    #  generate masses if necessary
    if m is None:
        nc = 1000 - 250 * ndim  # use nc cells for mass integration
        dx = boxsize / nc

        #  integrate total mass in box
        newparams = ic_sim_params.copy()
        newparams.nx = nc
        xc = IC_uniform_coordinates(newparams)
        rho_all = rho_anal(xc.value, ndim)
        if rho_all.any() < 0:
            raise ValueError(
                "Found negative densities inside box using the analytical function you provided"
            )
        rhotot = rho_all.sum()
        area = 1.0
        for d in range(ndim):
            area *= dx[d]
        mtot = rhotot * area  # go from density to mass

        m = np.ones(npart, dtype=np.float) * mtot / npart
        print("Assigning particle mass: {0:.3e}".format(mtot / npart))

    else:
        # switch to numpy arrays
        m = m.to(ic_sim_params.unit_m).value

    # empty arrays
    delta_r = np.zeros(x.shape, dtype=np.float)
    rho = np.zeros(npart, dtype=np.float)
    h = np.zeros(npart, dtype=np.float)
    hmodel = np.zeros(npart, dtype=np.float)

    # use unitless arrays from this point on
    x_nounit = x.value

    if ic_run_params.dumpfreq > 0:

        tree = KDTree(x_nounit[:, :ndim], boxsize=boxsize_for_tree)
        for p in range(npart):
            dist, neighs = tree.query(x_nounit[p, :ndim], k=Nngb_int)
            h[p] = dist[-1] / kernel_gamma
            for i, n in enumerate(neighs):
                W = kernel_func(dist[i], dist[-1])
                rho[p] += W * m[n]
        _IC_write_intermediate_output(0, x, m, rho, h, ic_sim_params, ic_run_params)
        # drop a first plot
        # TODO: remove the plotting
        _IC_plot_current_situation(True, 0, x_nounit, rho, rho_anal, ic_sim_params)

    # start iteration loop
    iteration = 0

    while True:

        iteration += 1

        # reset arrays
        rhoA = rho_anal(x_nounit, ndim)
        delta_r.fill(0.0)
        rho.fill(0.0)

        # re-distribute and/or dump particles?
        dump_now = ic_run_params.dumpfreq > 0
        dump_now = dump_now and iteration % ic_run_params.dumpfreq == 0
        redistribute = iteration % ic_run_params.redist_freq == 0
        redistribute = redistribute and ic_run_params.redist_stop >= iteration

        if dump_now or redistribute:

            # first build new tree
            tree = KDTree(x_nounit[:, :ndim], boxsize=boxsize_for_tree)
            for p in range(npart):
                dist, neighs = tree.query(x_nounit[p, :ndim], k=Nngb_int)
                h[p] = dist[-1] / kernel_gamma
                for i, n in enumerate(neighs):
                    W = kernel_func(dist[i], dist[-1])
                    rho[p] += W * m[n]

            if dump_now:
                _IC_write_intermediate_output(
                    iteration, x_nounit, m, rho, h, ic_sim_params, ic_run_params
                )
                # TODO: remove the plotting
                _IC_plot_current_situation(
                    True, iteration, x_nounit, rho, rho_anal, ic_sim_params
                )

            # re-destribute a handful of particles
            if redistribute:
                x_nounit, touched = redistribute_particles(
                    x_nounit, h, rho, rhoA, iteration, ic_sim_params, ic_run_params
                )
                # updated analytical density computations
                if touched is not None:
                    rhoA[touched] = rho_anal(x_nounit[touched], ndim)

        # compute MODEL smoothing lengths
        oneoverrho = 1.0 / rhoA
        oneoverrhosum = np.sum(oneoverrho)

        if ndim == 1:
            hmodel = 0.5 * Nngb * oneoverrho / oneoverrhosum
        elif ndim == 2:
            hmodel = np.sqrt(Nngb / np.pi * oneoverrho / oneoverrhosum)
        elif ndim == 3:
            hmodel = np.cbrt(Nngb * 3 / 4 / np.pi * oneoverrho / oneoverrhosum)

        # build tree, do neighbour loops
        tree = KDTree(x_nounit[:, :ndim], boxsize=boxsize_for_tree)

        for p in range(npart):

            dist, neighs = tree.query(x_nounit[p, :ndim], k=Nngb_int)
            # tree.query returns index npart where not enough neighbours are found
            correct = neighs < npart
            dist = dist[correct][1:]  # skip first neighbour: that's particle itself
            neighs = neighs[correct][1:]
            dx = x_nounit[p] - x_nounit[neighs]

            if periodic:
                for d in range(ndim):
                    boundary = boxsize[d]
                    bhalf = 0.5 * boundary
                    dx[dx[:, d] > bhalf, d] -= boundary
                    dx[dx[:, d] < -bhalf, d] += boundary

            for n, Nind in enumerate(neighs):  # skip 0: this is particle itself
                hij = (hmodel[p] + hmodel[Nind]) * 0.5
                Wij = kernel_func(dist[n], hij)
                delta_r[p] += hij * Wij / dist[n] * dx[n]

        if compute_delta_norm:
            # set initial delta_norm such that max displacement is 1 mean interparticle distance
            delrsq = np.zeros(npart, dtype=np.float)
            for d in range(ndim):
                delrsq += delta_r[:, d] ** 2
            delrsq = np.sqrt(delrsq)
            delta_r_norm = mid / delrsq.max()
            compute_delta_norm = False

        # finally, displace particles
        delta_r[:, :ndim] *= delta_r_norm
        x_nounit[:, :ndim] += delta_r[:, :ndim]

        # check whether something's out of bounds
        if periodic:
            for d in range(ndim):

                boundary = boxsize[d]

                xmax = 2 * boundary
                while xmax > boundary:
                    over = x_nounit[:, d] > boundary
                    x_nounit[over, d] -= boundary
                    xmax = x_nounit[:, d].max()

                xmin = -1.0
                while xmin < 0.0:
                    under = x_nounit[:, d] < 0.0
                    x_nounit[under, d] += boundary
                    xmin = x_nounit[:, d].min()
        else:
            # leave it where it was. This is a bit sketchy, better ideas are welcome.
            for d in range(ndim):
                boundary = boxsize[d]
                x_nounit[x_nounit > boundary] -= delta_r[x_nounit > boundary]
                x_nounit[x_nounit < 0.0] -= delta_r[x_nounit < 0.0]

        # reduce delta_r_norm
        delta_r_norm *= ic_run_params.delta_reduct
        # assert minimal delta_r
        delta_r_norm = max(delta_r_norm, delta_r_norm_min)

        # get displacements in units of mean interparticle distance
        dev = np.zeros(npart, dtype=np.float)
        for d in range(ndim):
            dev += delta_r[:, d] ** 2
        dev = np.sqrt(dev)
        dev /= mid

        max_deviation = dev.max()
        min_deviation = dev.min()
        av_deviation = dev.sum() / dev.shape[0]

        print(
            "Iteration {0:4d}; Displacement [mean interpart dist] Min: {1:8.5f} Average: {2:8.5f}; Max: {3:8.5f};".format(
                iteration, min_deviation, av_deviation, max_deviation
            )
        )

        if (
            max_deviation < ic_run_params.displ_thresh
        ):  # don't think about stopping until max < threshold
            unconverged = dev[dev > ic_run_params.converge_thresh.shape[0]]
            if unconverged < ic_run_params.tolerance_part * npart:
                print("Convergence criteria are met.")
                break

        if iteration == ic_run_params.iter_max:
            print("Reached max number of iterations without converging. Returning.")
            break

    coords = unyt_array(x_nounit, ic_sim_params.unit_l)
    masses = unyt_array(m, ic_sim_params.unit_m)
    stats = {}
    stats["min_motion"] = min_deviation
    stats["avg_motion"] = av_deviation
    stats["max_motion"] = max_deviation
    stats["niter"] = iteration

    return coords, masses, stats


def redistribute_particles(
    x: np.ndarray,
    h: np.ndarray,
    rho: np.ndarray,
    rhoA: np.ndarray,
    iteration: int,
    ic_sim_params: ic_sim_params,
    ic_run_params: ic_run_params,
):
    """
    Displace overdense particles into the proximity of underdense particles.


    Parameters
    -----------------

    x: np.ndarray
        numpy array of particle coordinates. Must have shape (npart, 3)

    h: np.ndarray
        numpy array of particle smoothing lengths

    rho: np.ndarray
        numpy array of SPH densities at particle positions

    rhoA: np.ndarray
        numpy array of the model (Analytical) density function evaluated at the 
        particle coordinates

    iteration: int
        current iteration of the initial condition generation algorithm
    
    ic_sim_params: ic_sim_params
        an ic_sim_params instance containing simulation parameters

    ic_run_params: ic_run_params
        an ic_run_params instance containing simulation parameters


    Returns
    -------------------

    x: np.ndarray
        numpy array of updated particle coordinates

    touched: np.ndarray or None
        indices of particles that have been moved around in this routine. If 
        ``None``, no particles have been moved.
    """

    npart = x.shape[0]
    boxsize = ic_sim_params.boxsize_to_use
    ndim = ic_sim_params.ndim

    # how many particles are we moving?
    to_move = int(npart * ic_run_params.redist_frac)
    to_move = max(to_move, 0.0)
    if to_move == 0:
        return x, None

    # decrease how many particles you move as number of iterations increases
    ic_run_params.redist_frac *= ic_run_params.redist_reduct

    _, _, kernel_gamma = get_kernel_data(ic_sim_params.kernel, ndim)

    underdense = rho < rhoA  # is this underdense particle?
    overdense = rho > rhoA  # is this overdense particle?
    touched = np.zeros(
        npart, dtype=np.bool
    )  # has this particle been touched as target or as to be moved?
    indices = np.arange(npart)  # particle indices

    moved = 0

    nover = overdense[overdense].shape[0]
    nunder = underdense[underdense].shape[0]
    if nover == 0 or nunder == 0:
        return x, None

    while moved < to_move:

        # pick an overdense random particle
        oind = indices[overdense][np.random.randint(0, nover)]
        if touched[oind]:
            continue  # skip touched particles

        # do we work with it?
        othresh = (rho[oind] - rhoA[oind]) / rho[oind]
        othresh = erf(othresh)
        if np.random.uniform() < othresh:

            attempts = 0
            while attempts < nunder:

                attempts += 1

                u = np.random.randint(0, nunder)
                uind = indices[underdense][u]

                if touched[uind]:
                    continue  # skip touched particles

                uthresh = (rhoA[uind] - rho[uind]) / rhoA[uind]

                if np.random.uniform() < uthresh:
                    # we have a match!
                    # compute displacement for overdense particle
                    dx = np.zeros(3, dtype=np.float)
                    H = kernel_gamma * h[uind]
                    for d in range(ndim):
                        sign = 1 if np.random.random() < 0.5 else -1
                        dx[d] = np.random.uniform() * 0.3 * H * sign

                    x[oind] = x[uind] + dx
                    touched[oind] = True
                    touched[uind] = True
                    moved += 1
                    break

    if moved > 0:

        # check boundary conditions
        if ic_sim_params.periodic:
            for d in range(ndim):
                x[x[:, d] > boxsize[d], d] -= boxsize[d]
                x[x[:, d] < 0.0, d] += boxsize[d]
        else:
            for d in range(ndim):

                # move them away from the edge by a random factor of mean "cell size" boxsize/npart^ndim
                mask = x[:, d] > boxsize[d]
                x[mask, d] = boxsize[d] * (
                    1.0 - np.random.uniform(x[x > 1.0].shape) / npart ** (1.0 / ndim)
                )

                mask = x[:, d] < 0.0
                x[mask, d] = (
                    np.random.uniform(x[mask, d].shape)
                    * boxsize[d]
                    / npart ** (1.0 / ndim)
                )

    return x, indices[touched]


def _IC_write_intermediate_output(
    iteration: int,
    x: np.ndarray,
    m: np.ndarray,
    rho: np.ndarray,
    h: np.ndarray,
    ic_sim_params: ic_sim_params,
    ic_run_params: ic_run_params,
):
    r"""
    Write an intermediate output of current particle positions, densities,
    masses, and smoothing lengths.

    Parameters
    ----------------

    iteration: int
        current iteration number

    x: np.ndarray
        current particle coordinates

    m: np.ndarray
        particle masses

    rho: np.ndarray
        particle densities

    h: np.ndarray
        particle smoothing lengths

    ic_sim_params: ic_sim_params
        an ic_sim_params instance containing simulation parameters

    ic_run_params: ic_run_params
        an ic_run_params instance containing simulation parameters


    Returns
    -------------------

    Nothing.

    """

    nx = ic_sim_params.nx
    ndim = ic_sim_params.ndim
    npart = nx ** ndim

    ICunits = unyt.UnitSystem(
        "IC_generation", ic_sim_params.unit_l, ic_sim_params.unit_m, unyt.s
    )

    W = Writer(ICunits, ic_sim_params.boxsize)
    W.gas.coordinates = unyt_array(x, ic_sim_params.unit_l)
    W.gas.smoothing_length = unyt_array(h, ic_sim_params.unit_l)
    W.gas.masses = unyt_array(m, ic_sim_params.unit_m)
    W.gas.densities = unyt_array(
        rho, W.gas.masses.units / W.gas.coordinates.units ** ndim
    )

    # invent some junk to fill up necessary arrays
    W.gas.internal_energy = unyt_array(
        np.zeros(npart, dtype=np.float), unyt.m ** 2 / unyt.s ** 2
    )
    W.gas.velocities = unyt_array(np.zeros(npart, dtype=np.float), unyt.m / unyt.s)

    # If IDs are not present, this automatically generates
    fname = ic_run_params.dump_basename + str(iteration).zfill(5) + ".hdf5"
    W.write(fname)

    return


def _IC_plot_current_situation(
    save: bool,
    iteration: int,
    x: np.ndarray,
    rho: np.ndarray,
    rho_anal: callable,
    ic_sim_params: dict,
):
    r"""
    Create a plot of what things look like now. In particular, scatter the
    particle positions and show what the densities currently look like.

    Parameters
    ----------------

    save: bool
        Whether to save to file (if True), or just show a plot.

    iteration: int 
        Current iteration number.

    x: np.ndarray 
        particle positions. Must be numpy array.

    rho: np.ndarray
        numpy array of SPH densities at particle positions

    rho_anal: callable
        The density function that is to be reproduced in the initial conditions.
        It must take two positional arguments:
        
        - ``x``: np.ndarray of 3D particle coordinates (even if your initial 
          conditions have lower dimensionality)
        - ``ndim``: integer, number of dimensions that your simulations is to 
          have
    
    ic_sim_params: dict
        a dict containing simulation parameters as returned by 
        ``IC_set_IC_params()``.

    Note
    ---------------

    + For debugging/checking purposes only, not meant to be called.
    """
    from matplotlib import pyplot as plt
    from scipy import stats

    boxsize = ic_sim_params.boxsize_to_use
    ndim = ic_sim_params.ndim
    nx = ic_sim_params.nx

    if x.shape[0] > 5000:
        marker = ","
    else:
        marker = "."

    # dict to generate coordinates to compute analytical solution
    tempparams = ic_sim_params.copy()
    tempparams.ndim = 1
    tempparams.nx = 200

    if ndim == 1:

        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax1 = fig.add_subplot(111)
        ax1.scatter(x[:, 0], rho, s=1, c="k", label="IC")

        XA = IC_uniform_coordinates(tempparams).value
        ax1.plot(XA[:, 0], rho_anal(XA, ndim), label="analytical")
        ax1.set_xlabel("x")
        ax1.set_ylabel("rho")
        ax1.legend()

    elif ndim == 2:

        fig = plt.figure(figsize=(18, 6), dpi=200)

        ax1 = fig.add_subplot(131, aspect="equal")
        ax2 = fig.add_subplot(132,)
        ax3 = fig.add_subplot(133,)

        # x - y scatterplot

        ax1.scatter(x[:, 0], x[:, 1], s=1, alpha=0.5, c="k", marker=marker)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # x - rho plot
        rho_mean, edges, _ = stats.binned_statistic(
            x[:, 0], rho, statistic="mean", bins=min(nx, 50)
        )
        rho2mean, edges, _ = stats.binned_statistic(
            x[:, 0], rho ** 2, statistic="mean", bins=min(nx, 50)
        )
        rho_std = np.sqrt(rho2mean - rho_mean ** 2)

        r = 0.5 * (edges[:-1] + edges[1:])
        ax2.errorbar(r, rho_mean, yerr=rho_std, label="average all ptcls", lw=1)

        tempparams.boxsize_to_use[0] = ic_sim_params.boxsize_to_use[0]
        XA = IC_uniform_coordinates(tempparams).value
        XA[:, 1] = 0.5 * boxsize[1]
        ax2.plot(XA[:, 0], rho_anal(XA, ndim), label="analytical, y = 0.5")

        ax2.scatter(
            x[:, 0], rho, s=1, alpha=0.4, c="k", label="all particles", marker=","
        )
        selection = np.logical_and(
            x[:, 1] > 0.45 * boxsize[1], x[:, 1] < 0.55 * boxsize[1]
        )
        ax2.scatter(
            x[selection, 0],
            rho[selection],
            s=2,
            alpha=0.8,
            c="r",
            label="0.45 boxsize < y < 0.55 boxsize",
        )

        ax2.legend()
        ax2.set_xlabel("x")
        ax2.set_ylabel("rho")

        # y - rho plot
        rho_mean, edges, _ = stats.binned_statistic(
            x[:, 1], rho, statistic="mean", bins=min(nx, 50)
        )
        rho2mean, edges, _ = stats.binned_statistic(
            x[:, 1], rho ** 2, statistic="mean", bins=min(nx, 50)
        )
        rho_std = np.sqrt(rho2mean - rho_mean ** 2)

        r = 0.5 * (edges[:-1] + edges[1:])
        ax3.errorbar(r, rho_mean, yerr=rho_std, label="average all ptcls", lw=1)

        tempparams.boxsize_to_use[0] = ic_sim_params.boxsize_to_use[1]
        XA = IC_uniform_coordinates(tempparams).value
        XA[:, 1] = XA[:, 0]
        XA[:, 0] = 0.5 * boxsize[0]
        xa = XA[:, 0]
        ax3.plot(XA[:, 1], rho_anal(XA, ndim), label="analytical x = 0.5")

        ax3.scatter(
            x[:, 1], rho, s=1, alpha=0.4, c="k", label="all particles", marker=","
        )
        selection = np.logical_and(
            x[:, 0] > 0.45 * boxsize[0], x[:, 0] < 0.55 * boxsize[0]
        )
        ax3.scatter(
            x[selection, 1],
            rho[selection],
            s=2,
            alpha=0.8,
            c="r",
            label="0.45 < y < 0.55",
        )

        ax3.legend()
        ax3.set_xlabel("y")
        ax3.set_ylabel("rho")

    elif ndim == 3:

        fig = plt.figure(figsize=(15, 10), dpi=200)

        ax1 = fig.add_subplot(231, aspect="equal")
        ax2 = fig.add_subplot(232, aspect="equal")
        ax3 = fig.add_subplot(233, aspect="equal")
        ax4 = fig.add_subplot(234,)
        ax5 = fig.add_subplot(235,)
        ax6 = fig.add_subplot(236,)

        # coordinate scatterplots

        ax1.scatter(x[:, 0], x[:, 1], s=1, alpha=0.5, c="k", marker=marker)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax2.scatter(x[:, 1], x[:, 2], s=1, alpha=0.5, c="k", marker=marker)
        ax2.set_xlabel("y")
        ax2.set_ylabel("z")

        ax3.scatter(x[:, 2], x[:, 0], s=1, alpha=0.5, c="k", marker=marker)
        ax3.set_xlabel("z")
        ax3.set_ylabel("x")

        # x - rho plot
        rho_mean, edges, _ = stats.binned_statistic(
            x[:, 0], rho, statistic="mean", bins=min(nx, 50)
        )
        rho2mean, edges, _ = stats.binned_statistic(
            x[:, 0], rho ** 2, statistic="mean", bins=min(nx, 50)
        )
        rho_std = np.sqrt(rho2mean - rho_mean ** 2)

        r = 0.5 * (edges[:-1] + edges[1:])
        ax4.errorbar(r, rho_mean, yerr=rho_std, label="average all ptcls", lw=1)

        tempparams.boxsize_to_use[0] = ic_sim_params.boxsize_to_use[0]
        XA = IC_uniform_coordinates(tempparams).value
        XA[:, 1] = 0.5 * boxsize[1]
        XA[:, 2] = 0.5 * boxsize[2]
        ax4.plot(XA[:, 0], rho_anal(XA, ndim), label="analytical, y, z = 1/2 boxsize")

        ax4.scatter(
            x[:, 0], rho, s=1, alpha=0.4, c="k", label="all particles", marker=","
        )

        ax4.legend()
        ax4.set_xlabel("x")
        ax4.set_ylabel("rho")

        # y - rho plot
        rho_mean, edges, _ = stats.binned_statistic(
            x[:, 1], rho, statistic="mean", bins=min(50, nx)
        )
        rho2mean, edges, _ = stats.binned_statistic(
            x[:, 1], rho ** 2, statistic="mean", bins=min(50, nx)
        )
        rho_std = np.sqrt(rho2mean - rho_mean ** 2)

        r = 0.5 * (edges[:-1] + edges[1:])
        ax5.errorbar(r, rho_mean, yerr=rho_std, label="average all ptcls", lw=1)

        tempparams.boxsize_to_use[0] = ic_sim_params.boxsize_to_use[1]
        XA = IC_uniform_coordinates(tempparams).value
        XA[:, 1] = XA[:, 0]
        XA[:, 0] = 0.5 * boxsize[0]
        XA[:, 2] = 0.5 * boxsize[2]
        ax5.plot(XA[:, 1], rho_anal(XA, ndim), label="analytical, x, z = 1/2 boxsize")

        ax5.scatter(
            x[:, 1], rho, s=1, alpha=0.4, c="k", label="all particles", marker=","
        )

        ax5.legend()
        ax5.set_xlabel("y")
        ax5.set_ylabel("rho")

        # z - rho plot
        rho_mean, edges, _ = stats.binned_statistic(
            x[:, 2], rho, statistic="mean", bins=min(50, nx)
        )
        rho2mean, edges, _ = stats.binned_statistic(
            x[:, 2], rho ** 2, statistic="mean", bins=min(50, nx)
        )
        rho_std = np.sqrt(rho2mean - rho_mean ** 2)

        r = 0.5 * (edges[:-1] + edges[1:])
        ax6.errorbar(r, rho_mean, yerr=rho_std, label="average all ptcls", lw=1)

        tempparams.boxsize_to_use[0] = ic_sim_params.boxsize_to_use[2]
        XA = IC_uniform_coordinates(tempparams).value
        XA[:, 2] = XA[:, 0]
        XA[:, 0] = 0.5 * boxsize[0]
        XA[:, 1] = 0.5 * boxsize[2]
        ax6.plot(XA[:, 2], rho_anal(XA, ndim), label="analytical, x, y = 1/2 boxsize")

        ax6.scatter(
            x[:, 2], rho, s=1, alpha=0.4, c="k", label="all particles", marker=","
        )

        ax6.legend()
        ax6.set_xlabel("z")
        ax6.set_ylabel("rho")

    ax1.set_title("Iteration =" + str(iteration))
    if save:
        plt.savefig("plot-IC-generation-iteration-" + str(iteration).zfill(5) + ".png")
        print("plotted current situation and saved figure to file.")
    else:
        plt.show()
    plt.close()

    return
