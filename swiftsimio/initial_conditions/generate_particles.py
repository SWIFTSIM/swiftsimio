"""Generate SPH initial conditions for SPH simulations iteratively for a given density function following Arth et al.
2019 (https://arxiv.org/abs/1907.11250).

"""
import numpy as np
import unyt
from math import erf
from typing import Union
import warnings

from .IC_kernel import get_kernel_data
from swiftsimio.optional_packages import KDTree, TREE_AVAILABLE
from swiftsimio import Writer


class RunParams(object):
    r"""
    TODO: dox


    Attributes
    ---------------
    iter_max: int, optional
        max numbers of iterations for generating IC conditions

    converge_thresh: float, optional
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

    Tdump_basename: str
        Basename for intermediate dumps. The filename will be constructed as
        ``basename + <5 digit zero padded iteration number> + .hdf5``

    random_seed: int, optional
        set a specific random seed

    """
    # iteration parameters
    iter_max: int = 10000
    iter_min: int = 0
    converge_thresh: float = 1e-4
    tolerance_part: float = 1e-3
    displ_thresh: float = 1e-3
    delta_init: Union[float, None] = None
    delta_reduct: float = 1.0
    delta_min: float = 1e-6
    redist_freq: int = 20
    redist_frac: float = 0.01
    redist_reduct: float = 1.0
    redist_stop: int = 200
    dump_basename: str = "IC-generation-iteration-"
    dumpfreq: int = 50

    def __init__(self):
        # always set one random seed.
        self.set_random_seed(666)
        return

    def set_random_seed(self, seed: int = 666):
        r"""
        TODO: dox
        """
        np.random.seed(seed)
        return


class IterData(object):
    r"""
    TODO: dox
    contains data relevant during the iteration
    """

    mean_interparticle_distance: float = 1.0
    delta_r_norm: float = 1.0
    delta_r_norm_min: float = 1e-2
    compute_delta_r_norm: bool = True
    Nngb: float = 20.0
    Nngb_int: int = 20
    boxsize_for_tree: np.ndarray = np.ones(3, dtype=float)

    def __init__(self):
        return


class IterStats(object):
    r"""
    used to store stats of the iteration.
    """

    max_displacement: np.ndarray
    min_displacement: np.ndarray
    avg_displacement: np.ndarray
    niter: int

    def __init__(self, iter_max):
        self.max_displacement = np.zeros(iter_max, dtype=np.float)
        self.min_displacement = np.zeros(iter_max, dtype=np.float)
        self.avg_displacement = np.zeros(iter_max, dtype=np.float)
        self.niter = 0
        return

    def add_stat(
        self,
        iteration,
        min_displacement: float,
        max_displacement: float,
        avg_displacement: float,
    ):
        r"""
        Store new stats at the appropriate place.
        """

        self.max_displacement[iteration] = max_displacement
        self.min_displacement[iteration] = min_displacement
        self.avg_displacement[iteration] = avg_displacement
        self.niter = iteration
        return


class ParticleGenerator(object):
    r"""

    TODO: dox

    Set up the simulation parameters for the initial conditions you want to
    generate. The dict it returns is a necessary argument to call
    ``generate_IC_for_given_density()``


    Parameters
    -------------

    rho: callable
        The density function that is to be reproduced in the initial conditions.
        It must take two positional arguments:
        
        - ``x``: np.ndarray of 3D particle coordinates (even if your initial 
          conditions have lower dimensionality)
        - ``ndim``: integer, number of dimensions that your simulations is to 
          have

    boxsize: unyt.unyt_array
        The box size of the simulation.

    nx: int
        how many particles along every dimension you want your simulation
        to contain

    ndim: int
        how many dimensions you want your simulation to have

    unitsys: unyt.unit_systems.UnitSystem
        a unit system that contains the units you want your ICs to have

    periodic: bool, optional
        whether the simulation box is periodic or not

    rho_max: float or None, optional
        The maximal density within the simulation box. If ``None``, an 
        approximate value will be determined if the rejection sampling to obtain
        an initial particle configuration guess is used.

    kernel: str {'cubic spline',}, optional
        which kernel to use

    eta: float, optional
        resolution eta, which defines the number of neighbours used
        independently of dimensionality



    Notes
    -----------
    
    + The returned dict is a required argument to call ``generate_IC_for_given_density()``
    """

    # simulation parameters
    rhofunc: callable
    boxsize: unyt.unyt_array
    nx: int
    ndim: int
    periodic: bool = True
    unitsys: unyt.unit_systems.UnitSystem
    kernel: str = "cubic spline"
    eta: float = 1.2348
    rhomax: Union[float, None] = None

    runparams: RunParams = RunParams()
    iterparams: IterData = IterData()
    stats: IterStats

    # derived variables
    npart = 0

    # internal checks
    _set_up = False

    # unyt/result arrays
    coordinates: Union[unyt.unyt_array, None] = None
    masses: Union[unyt.unyt_array, None] = None
    smoothing_lengths: Union[unyt.unyt_array, None] = None
    densities: Union[unyt.unyt_array, None] = None

    # unitless arrays to work with
    x: np.array
    m: np.array

    def __init__(
        self,
        rho: callable,
        boxsize: unyt.unyt_array,
        unitsys: unyt.unit_systems.UnitSystem,
        nx: int,
        ndim: int,
        periodic: bool = True,
        rho_max: Union[float, None] = None,
        kernel: str = "cubic spline",
        eta: float = 1.2348,
    ):

        if not isinstance(boxsize, unyt.unyt_array):
            raise TypeError("boxsize needs to be a unyt array.")

        if not isinstance(unitsys, unyt.unit_systems.UnitSystem):
            raise TypeError("unitsys needs to be a unyt UnitSystem.")

        if not isinstance(nx, int):
            raise TypeError("nx needs to be an integer")

        if not isinstance(ndim, int):
            raise TypeError("ndim needs to be an integer")

        # TODO: order!
        self.rhofunc = rho
        self.boxsize = boxsize
        self.unitsys = unitsys
        self.rho_max = rho_max
        self.periodic = periodic
        self.nx = nx
        self.ndim = ndim
        self.kernel = kernel
        self.eta = eta

        # get some derived quantities
        self.npart = self.nx ** self.ndim
        self.boxsize_to_use = boxsize.to(unitsys["length"]).value

        return

    def initial_setup(
        self,
        method: str = "rejection",
        x: Union[unyt.unyt_array, None] = None,
        m: Union[unyt.unyt_array, None] = None,
        max_displ: float = 0.4,
    ):
        r"""
        TODO: dox
        x: unyt.unyt_array or None, optional
            Initial guess for coordinates of particles. If ``None``, the initial
            guess will be generated by rejection sampling the density function
            ``rhofunc``

        m: unyt.unyt_array or None, optional
            ``unyt.unyt_array`` of particle masses. If ``None``, an array will be created
            such that the total mass in the simulation box given the analytical
            density is reproduced, and all particles will have equal masses.

        max_displ: float
            maximal displacement of a particle initially on an uniform grid along 
            any axis, in units of particle distance along that axis. Is only used
            if ``method = 'displaced'``
        """

        ndim = self.ndim
        rhofunc = self.rhofunc
        boxsize = self.boxsize_to_use
        npart = self.npart
        ip = self.iterparams
        runparams = self.runparams

        # safety checks first
        if not TREE_AVAILABLE:
            raise ImportError(
                "The scipy.spatial.cKDTree class is required to generate initial conditions."
            )

        try:
            res = self.rhofunc(np.ones((10, 3), dtype=np.float), self.ndim)
        except TypeError:
            raise TypeError(
                "rho(x, ndim) must take only a numpy array x of coordinates as an argument."
            )
        if not isinstance(res, np.ndarray):
            raise TypeError("rho(x, ndim) needs to return a numpy array as the result.")

        # generate first positions if necessary
        if x is None:
            if method == "rejection":
                self.coordinates = self.IC_sample_coordinates()
            elif method == "displaced":
                self.coordinates = self.IC_perturbed_coordinates(max_displ=max_displ)
            elif method == "uniform":
                self.coordinates = self.IC_uniform_coordinates()
            else:
                raise ValueError("Unknown coordinate generation method:", method)
        else:
            if not isinstance(x, unyt.unyt_array):
                raise TypeError("x must be an unyt array")
            # TODO: compare given x array units with unit system first
            self.coordinates = x

        #  generate masses if necessary
        if m is None:
            nc = int(10000 ** (1.0 / ndim) + 0.5)  # always use ~ 10000 mesh points
            dx = boxsize / nc

            #  integrate total mass in box
            xc = self.IC_uniform_coordinates(nx=nc)
            rho_all = rhofunc(xc.value, ndim)
            if rho_all.any() < 0:
                raise ValueError(
                    "Found negative densities inside box using the analytical function you provided"
                )
            rhotot = rho_all.sum()
            area = 1.0
            for d in range(ndim):
                area *= dx[d]
            mtot = rhotot * area  # go from density to mass

            self.m = np.ones(npart, dtype=np.float) * mtot / npart
            self.masses = unyt.unyt_array(self.m, self.unitsys["mass"])
            print("Assigning particle mass: {0:.3e}".format(mtot / npart))

        else:
            # TODO: compare given mass array units with unit system first
            if not isinstance(m, unyt.unyt_array):
                raise TypeError("m must be an unyt array")
            self.masses = m.to(self.unitsys["mass"]).value

        # make sure unitless arrays exist, others are allocated
        self.x = self.coordinates.value
        self.m = self.masses.value

        # check consistency of runtime params

        if runparams.iter_max < runparams.iter_min:
            raise ValueError("runparams.iter_max must be >= runparams.iter_min")
        if runparams.delta_init is not None:
            if runparams.delta_init < runparams.delta_min:
                raise ValueError("runparams.delta_init must be >= runparams.delta_min")

        # set up iteration related stuff

        # get mean interparticle distance
        mid = 1.0
        for d in range(ndim):
            mid *= boxsize[d]
        mid = mid ** (1.0 / ndim) / self.nx
        ip.mean_interparticle_distance = mid

        # get normalisation constants for displacement force
        if runparams.delta_init is None:
            ip.compute_delta_norm = True
            ip.delta_r_norm = mid
        else:
            ip.compute_delta_norm = False
            ip.delta_r_norm = runparams.delta_init * mid

        ip.delta_r_norm_min = runparams.delta_min * mid

        # kernel data
        _, _, kernel_gamma = get_kernel_data(self.kernel, ndim)

        # get expected number of neighbours
        if ndim == 1:
            ip.Nngb = 1 * kernel_gamma * self.eta
        elif ndim == 2:
            ip.Nngb = np.pi * (kernel_gamma * self.eta) ** 2
        elif ndim == 3:
            ip.Nngb = 4 / 3 * np.pi * (kernel_gamma * self.eta) ** 3

        # round it up for cKDTree
        ip.Nngb_int = int(ip.Nngb + 0.5)

        #  this sets up whether the tree build is periodic or not
        if self.periodic:
            ip.boxsize_for_tree = self.boxsize_to_use[:ndim]
        else:
            ip.boxsize_for_tree = None

        # set up stats
        self.stats = IterStats(self.runparams.iter_max)

        # take note that we did this
        self._set_up = True

        return

    def IC_uniform_coordinates(
        self, nx: Union[None, int] = None, ndim: Union[None, int] = None
    ):
        r"""
        Generate coordinates for a uniform particle distribution.
        TODO: fix dox

        Parameters
        ------------------


        Returns
        ------------------

        x: unyt.unyt_array 
            unyt.unyt_array(shape=(npart, 3), dtype=float) of coordinates, where 
            ``npart = nx ** ndim``, both of which are set in ``self``
        """
        if nx is None:
            nx = self.nx
        if ndim is None:
            ndim = self.ndim
        boxsize = self.boxsize_to_use

        # get npart here, nx and ndim might be different from global class values
        npart = nx ** ndim

        x = unyt.unyt_array(
            np.zeros((npart, 3), dtype=np.float), self.unitsys["length"]
        )

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

    def IC_perturbed_coordinates(self, max_displ: float = 0.4):
        """
        Get the coordinates for a randomly perturbed uniform particle distribution.
        The perturbation won't exceed ``max_displ`` times the interparticle distance
        along an axis.

        TODO: check dox


        Parameters
        ------------------

        max_displ: float
            maximal displacement of a particle initially on an uniform grid along 
            any axis, in units of particle distance along that axis


        Returns
        ------------------

        x: unyt.unyt_array 
            unyt.unyt_array(shape=(npart, 3), dtype=float) of coordinates, where 
            ``npart = nx ** ndim``, both of which are set in the ``self`` 
        """

        nx = self.nx
        boxsize = self.boxsize_to_use
        ndim = self.ndim
        periodic = self.periodic
        npart = nx ** ndim

        # get maximal displacement from uniform grid of any particle along an axis
        maxdelta = max_displ * boxsize / nx

        # generate uniform grid (including units) first
        x = self.IC_uniform_coordinates()

        for d in range(ndim):
            amplitude = unyt.unyt_array(
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
                    amplitude_redo = unyt.unyt_array(
                        np.random.uniform(low=-boxsize[d], high=boxsize[d], size=nredo)
                        * maxdelta[d],
                        x.units,
                    )
                    x[redo, d] += amplitude_redo

                    xmax = x[:, d].max()
                    xmin = x[:, d].min()

        return x

    def IC_sample_coordinates(self):
        r"""
        Generate an initial guess for particle coordinates by rejection sampling the
        density


        Parameters
        ------------------


        Returns
        ------------------

        x: unyt.unyt_array 
            unyt.unyt_array(shape=(npart, 3), dtype=float) of coordinates, where 
            ``npart = nx ** ndim``, both of which are set in ``self`` 
        """

        boxsize = self.boxsize_to_use
        ndim = self.ndim
        npart = self.npart
        rhofunc = self.rhofunc

        x = np.empty((npart, 3), dtype=np.float)

        if self.rho_max is None:
            # find approximate peak value of rho_max
            # don't cause memory errors with too big of a grid.
            #  Also don't worry too much about accuracy.
            nc = 200
            xc = self.IC_uniform_coordinates(nx=nc)
            self.rho_max = rhofunc(xc.value, ndim).max() * 1.05
            # * 1.05: safety measure to make sure you're always above the
            #  analytical value. Takes a tad more time, but we're gonna be safe.

        keep = 0
        coord_threshold = boxsize
        while keep < npart:

            xr = np.zeros((1, 3), dtype=np.float)
            for d in range(ndim):
                xr[0, d] = np.random.uniform(low=0.0, high=coord_threshold[d])

            if np.random.uniform() <= rhofunc(xr, ndim) / self.rho_max:
                x[keep] = xr
                keep += 1

        return unyt.unyt_array(x, self.unitsys["length"])

    def compute_h_and_rho(
        self, nngb: Union[int, None] = None, boxsize: Union[np.ndarray, None] = None
    ):
        r"""
        TODO: DOCS
        make sure to call self.initial_setup() first to allocate arrays

        Parameters
        -----------------

        boxsize: None or np.ndarray

        nngb: integer
        """
        if nngb is None:
            nngb = self.iterparams.Nngb_int
        if boxsize is None:
            boxsize = self.iterparams.boxsize_for_tree

        ndim = self.ndim
        kernel_func, _, kernel_gamma = get_kernel_data(self.kernel, ndim)

        rho = np.zeros(self.npart, dtype=np.float)
        h = np.zeros(self.npart, dtype=np.float)

        tree = KDTree(self.x[:, :ndim], boxsize=boxsize)
        for p in range(self.npart):
            dist, neighs = tree.query(self.x[p, :ndim], k=nngb)
            h[p] = dist[-1] / kernel_gamma
            for i, n in enumerate(neighs):
                W = kernel_func(dist[i], dist[-1])
                rho[p] += W * self.m[n]

        return h, rho

    def perform_iteration(
        self, iteration: int,
    ):
        """
        TODO: DOX
        """
        ndim = self.ndim
        periodic = self.periodic
        boxsize = self.boxsize_to_use
        npart = self.npart
        rhofunc = self.rhofunc
        x = self.x
        runparams = self.runparams
        ipars = self.iterparams

        from .IC_plotting import IC_plot_current_situation

        # kernel data
        kernel_func, _, kernel_gamma = get_kernel_data(self.kernel, ndim)

        # update model density at current particle positions
        rho_model = rhofunc(x, ndim)

        # re-distribute and/or dump particles?
        dump_now = runparams.dumpfreq > 0
        dump_now = dump_now and iteration % runparams.dumpfreq == 0
        redistribute = iteration % runparams.redist_freq == 0
        redistribute = redistribute and runparams.redist_stop >= iteration

        if dump_now or redistribute:

            # first build new tree, get smoothing lengths and densities
            h, rho = self.compute_h_and_rho()

            if dump_now:
                self.IC_write_intermediate_output(iteration, h, rho)
                # TODO: remove the plotting
                IC_plot_current_situation(True, iteration, x, rho, rhofunc, self)

            # re-destribute a handful of particles
            if redistribute:
                moved = self.redistribute_particles(h, rho, rho_model)
                #  update analytical density computations
                if moved is not None:
                    rho_model[moved] = rhofunc(x[moved], ndim)

        # compute MODEL smoothing lengths
        oneoverrho = 1.0 / rho_model
        oneoverrhosum = np.sum(oneoverrho)
        vol = 0.0
        for d in range(ndim):
            vol *= boxsize[d]

        if ndim == 1:
            hmodel = 0.5 * ipars.Nngb * oneoverrho / oneoverrhosum
            #  hmodel = 0.5 * ipars.Nngb * oneoverrho / oneoverrhosum * vol
        elif ndim == 2:
            hmodel = np.sqrt(ipars.Nngb / np.pi * oneoverrho / oneoverrhosum)
            #  hmodel = np.sqrt(ipars.Nngb / np.pi * oneoverrho / oneoverrhosum * vol)
        else:
            hmodel = np.cbrt(ipars.Nngb * 3 / 4 / np.pi * oneoverrho / oneoverrhosum)
            #  hmodel = np.cbrt(ipars.Nngb * 3 / 4 / np.pi * oneoverrho / oneoverrhosum * vol)

        # init delta_r array
        delta_r = np.zeros(self.x.shape, dtype=np.float)

        # build tree, do neighbour loops
        tree = KDTree(x[:, :ndim], boxsize=ipars.boxsize_for_tree)

        for p in range(npart):

            dist, neighs = tree.query(x[p, :ndim], k=ipars.Nngb_int)
            # tree.query returns index npart where not enough neighbours are found
            correct = neighs < npart
            dist = dist[correct][1:]  # skip first neighbour: that's particle itself
            neighs = neighs[correct][1:]
            dx = x[p] - x[neighs]

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

        if ipars.compute_delta_norm:
            # set initial delta_norm such that max displacement is
            # = 1 mean interparticle distance
            delrsq = np.zeros(npart, dtype=np.float)
            for d in range(ndim):
                delrsq += delta_r[:, d] ** 2
            delrsq = np.sqrt(delrsq)
            ipars.delta_r_norm = ipars.mean_interparticle_distance / delrsq.max()
            #  ipars.delta_r_norm *= 1e-2
            ipars.compute_delta_norm = False

        # finally, displace particles
        delta_r[:, :ndim] *= ipars.delta_r_norm
        x[:, :ndim] += delta_r[:, :ndim]

        # check whether something's out of bounds
        if periodic:
            for d in range(ndim):

                boundary = boxsize[d]

                xmax = 2 * boundary
                while xmax > boundary:
                    over = x[:, d] > boundary
                    x[over, d] -= boundary
                    xmax = x[:, d].max()

                xmin = -1.0
                while xmin < 0.0:
                    under = x[:, d] < 0.0
                    x[under, d] += boundary
                    xmin = x[:, d].min()

        else:

            # leave it where it was. This is a bit sketchy, better ideas are welcome.
            for d in range(ndim):
                boundary = boxsize[d]
                x[x > boundary] -= delta_r[x > boundary]
                x[x < 0.0] -= delta_r[x < 0.0]

        # reduce delta_r_norm
        ipars.delta_r_norm *= runparams.delta_reduct
        # assert minimal delta_r
        ipars.delta_r_norm = max(ipars.delta_r_norm, ipars.delta_r_norm_min)

        # get displacements in units of mean interparticle distance
        displacement = np.zeros(npart, dtype=np.float)
        for d in range(ndim):
            displacement += delta_r[:, d] ** 2
        displacement = np.sqrt(displacement)

        max_displacement = displacement.max()
        max_displacement /= ipars.mean_interparticle_distance
        min_displacement = displacement.min()
        min_displacement /= ipars.mean_interparticle_distance
        avg_displacement = displacement.sum() / displacement.shape[0]
        avg_displacement /= ipars.mean_interparticle_distance

        print(
            "Iteration {0:4d}; Min: {1:8.5f} Average: {2:8.5f}; Max: {3:8.5f};".format(
                iteration, min_displacement, avg_displacement, max_displacement
            )
        )

        if max_displacement > 5.0:
            # get the initial value. If delta_init was None, as is default, you
            # need to know what value to start with in your next run.
            dinit = (
                ipars.delta_r_norm
                * runparams.delta_reduct ** (-max(iteration, 1))
                / ipars.mean_interparticle_distance
            )
            warnmsg = "Found max displacements > 5 mean interparticle distances. "
            warnmsg += "Maybe try again with smaller runparams.delta_init? "
            warnmsg += "runparams.delta_init = {0:.6e}".format(dinit)
            warnings.warn(warnmsg, RuntimeWarning)

        converged = False
        if (
            max_displacement < runparams.displ_thresh
        ):  # don't think about stopping until max < threshold
            unconverged = displacement[displacement > runparams.converge_thresh].shape[
                0
            ]
            if unconverged < runparams.tolerance_part * npart:
                converged = True

        # store stats
        self.stats.add_stat(
            iteration, min_displacement, max_displacement, avg_displacement
        )

        return converged

    def generate_IC_for_given_density(self):
        """
        Generate SPH initial conditions for SPH simulations iteratively for a given
        density function ``rhofunc()`` following Arth et al. 2019 
        (https://arxiv.org/abs/1907.11250).

        TODO: Check dox


        Parameters
        ------------------



        Returns
        -----------

        """
        # this import is only temporary to avoid circular imports.
        # it will be removed before the merge.
        from .IC_plotting import IC_plot_current_situation

        if not self._set_up:
            self.initial_setup()

        # start iteration loop
        iteration = 0

        while iteration < self.runparams.iter_max:

            converged = self.perform_iteration(iteration)
            iteration += 1

            if converged and self.runparams.iter_min < iteration:
                break

        # convert results to unyt arrays
        self.coords = unyt.unyt_array(self.x, self.unitsys["length"])
        self.masses = unyt.unyt_array(self.m, self.unitsys["mass"])

        # compute densities and smoothing lengths before you finish
        h, rho = self.compute_h_and_rho()
        self.smoothing_lengths = unyt.unyt_array(h, self.unitsys["length"])
        self.densities = unyt.unyt_array(
            rho, self.unitsys["mass"] / self.unitsys["length"] ** self.ndim
        )

        # trim stats
        self.stats.max_displacement = self.stats.max_displacement[: self.stats.niter]
        self.stats.min_displacement = self.stats.min_displacement[: self.stats.niter]
        self.stats.avg_displacement = self.stats.avg_displacement[: self.stats.niter]

        return

    def redistribute_particles(
        self, h: np.ndarray, rho: np.ndarray, rhoA: np.ndarray,
    ):
        """
        Displace overdense particles into the proximity of underdense particles.


        Parameters
        -----------------

        rho: np.ndarray
            numpy array of particle smoothing lenghts

        rho: np.ndarray
            numpy array of SPH densities at particle positions

        rhoA: np.ndarray
            numpy array of the model (Analytical) density function evaluated at the 
            particle coordinates


        Returns
        -------------------

        moved: np.ndarray or None
            indices of particles that have been moved around in this routine. If 
            ``None``, no particles have been moved.
        """

        x = self.x

        npart = x.shape[0]
        boxsize = self.boxsize_to_use
        ndim = self.ndim

        # how many particles are we moving?
        to_move = int(npart * self.runparams.redist_frac + 0.5)
        to_move = max(to_move, 0)
        if to_move == 0:
            return None

        # decrease how many particles you move as number of iterations increases
        self.runparams.redist_frac *= self.runparams.redist_reduct

        _, _, kernel_gamma = get_kernel_data(self.kernel, ndim)

        indices = np.arange(npart)  # particle indices
        underdense = rho < rhoA  # is this underdense particle?
        overdense = rho > rhoA  # is this overdense particle?

        # has this particle been touched as target or as to be moved?
        touched = np.zeros(npart, dtype=np.bool)

        # indices of particles that have been moved
        moved = np.empty(to_move, dtype=np.int)

        nmoved = 0

        nover = overdense[overdense].shape[0]
        nunder = underdense[underdense].shape[0]
        if nover == 0 or nunder == 0:
            return None

        max_attempts_over = 10 * to_move  # only try your luck, don't force
        attempts_over = 0

        while nmoved < to_move and attempts_over < max_attempts_over:

            attempts_over += 1

            # pick an overdense random particle
            oind = indices[overdense][np.random.randint(0, nover)]
            if touched[oind]:
                continue  # skip touched particles

            # do we work with it?
            othresh = (rho[oind] - rhoA[oind]) / rho[oind]
            othresh = erf(othresh)
            if np.random.uniform() < othresh:

                attempts_under = 0
                while attempts_under < nunder:  # only try your luck, don't force

                    attempts_under += 1

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
                        moved[nmoved] = oind
                        nmoved += 1
                        break

        print("Moved:", nmoved)
        if nmoved > 0:

            # check boundary conditions
            if self.periodic:
                for d in range(ndim):
                    x[x[:, d] > boxsize[d], d] -= boxsize[d]
                    x[x[:, d] < 0.0, d] += boxsize[d]
            else:
                temp = 1.0 / npart ** (1.0 / ndim)
                for d in range(ndim):

                    # move them away from the edge by a random factor of mean "cell size" boxsize/npart^ndim
                    mask = x[:, d] > boxsize[d]
                    x[mask, d] = boxsize[d] * (
                        1.0 - np.random.uniform(x[mask].shape) * temp
                    )

                    mask = x[:, d] < 0.0
                    x[mask, d] = np.random.uniform(x[mask, d].shape) * boxsize[d] * temp

        return moved[:nmoved]

    def IC_write_intermediate_output(
        self, iteration: int, h: np.ndarray, rho: np.ndarray,
    ):
        r"""
        Write an intermediate output of current particle positions, densities,
        masses, and smoothing lengths.

        Parameters
        ----------------

        iteration: int
            current iteration number

        h: np.ndarray
            particle smoothing lengths

        rho: np.ndarray
            particle densities

        Returns
        -------------------

        Nothing.

        """
        ndim = self.ndim
        npart = self.npart
        x = self.x
        m = self.m

        uL = self.unitsys["length"]
        uM = self.unitsys["mass"]
        uT = self.unitsys["time"]

        W = Writer(self.unitsys, self.boxsize)
        W.gas.coordinates = unyt.unyt_array(x, uL)
        W.gas.smoothing_length = unyt.unyt_array(h, uL)
        W.gas.masses = unyt.unyt_array(m, uM)
        W.gas.densities = unyt.unyt_array(rho, uM / uL ** ndim)

        # invent some junk to fill up necessary arrays
        W.gas.internal_energy = unyt.unyt_array(
            np.ones(npart, dtype=np.float), uL ** 2 / uT ** 2
        )
        W.gas.velocities = unyt.unyt_array(np.zeros(npart, dtype=np.float), uL / uT)

        # If IDs are not present, this automatically generates
        fname = self.runparams.dump_basename + str(iteration).zfill(5) + ".hdf5"
        W.write(fname)

        return
