"""
Generate SPH initial conditions for SPH simulations iteratively for a given density function following Arth et al.
2019 (https://arxiv.org/abs/1907.11250).

"""
import numpy as np
import unyt
from math import erf
from typing import Union
import warnings
import h5py

from .IC_kernel import get_kernel_data
from swiftsimio.optional_packages import KDTree, TREE_AVAILABLE
from swiftsimio import Writer, load


class RunParams(object):
    r"""
    A class to store iteration runtime parameters. Before running the iteration,
    these parameters are intended to be direcly accessed by the user.


    Attributes
    ---------------
    self.iter_max: int
        maximal numbers of iterations to run for generating IC conditions

    self.iter_min: int
        minimal numbers of iterations to run for generating IC conditions

    self.converge_thresh: float
        upper threshold for what displacement is supposed to be considered as
        converged. If enough particles are displaced by distance below
        ``self.converge_thresh * mean interparticle distance``, stop iterating.
        ``self.tolerance_part`` defines, what "enough particles" is.

    self.tolerance_part: float
        tolerance for not converged particle fraction: this fraction of
        particles can be displaced with distances > self.converge_thresh

    self.displ_thresh: float
        displacement threshold. It's an iteration continuation criterion: Don't
        stop until every particle is displaced by a distance < `` self.displ_thresh
        * mean interparticle distance``

    self.delta_init: float or None
        initial normalization constant for particle motion in units of mean
        interparticle distance. If ``None`` (default), ``self.delta_init`` will be set
        such that the maximal displacement found in the first iteration is
        normalized to 1 mean interparticle distance.

    self.delta_reduct: float
        normalization constant reduction factor. Intended to be > 0 and < 1.
        Multiply the normalization constant for particle motion by this factor
        after every iteration. In certain difficult cases this might help the
        generation to converge if set to < 1.

    self.delta_min: float
        minimal normalization constant for particle motion in units of mean
        interparticle distance.

    self.redist_freq: int
        particle redistribution frequency.
        Redistribute a handful of particles every ``self.redistribute_frequency``
        iteration. How many particles are redistributed is controlled with the
        ``self.redistribute_frac`` parameter. If = 0, no redistributions will be
        performed.

    self.redist_frac: float
        fraction of particles to be redistributed when doing so.

    self.redist_reduct: float
        multiply the ``self.redistribute_fraction`` parameter by this factor every
        time particles are being redistributed. In certain difficult cases this
        might help the generation to converge if set to < 1.

    self.redist_stop: int
        don't redistribute particles after this many iterations.

    self.dumpfreq: int
        frequency of dumps of the current state of the iteration. If set to zero,
        no intermediate results will be stored. If > 0, it will also create a 
        dump after the last iteration so a restart is possible.

    self.dump_basename: str
        Basename for intermediate dumps. The filename will be constructed as
        ``basename + <5 digit zero padded iteration number> + .hdf5``

    self.random_seed: int, optional
        set a specific random seed



    Methods
    ----------

    self.set_random_seed(seed: int)
        set the random seed to whatever pleases you.

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
        Set the random seed to whatever pleases you.

        Parameters
        --------------

        seed: int, optional
            seed to use.
        """
        np.random.seed(seed)
        return


class IterData(object):
    r"""
    contains data relevant during the iteration. This data is
    automatically set up, and no user is supposed to tinker
    with this data during/before the iteration.

    Attributes
    -------------

    self.mean_interparticle_distance: float
        the mean interparticle distance

    self.delta_r_norm: float
        norm to be used to compute the displacement "force"

    self.delta_r_norm_min: float
        lower threshold for "force" normalisation.

    self.compute_delta_r_norm: bool
        whether the user wants the code to compute an appropriate
        normalisation.

    self.Nngb: float
        number of neighbours to be searched for

    self.Nngb_int: int
        number of neighbours to be searched for. Integer, to be used
        as argument for scipy.spatial.cKDTree.query().

    self.boxsize_for_tree: None or np.ndarray
        boxsize parameter to be used as argument for scipy.spatioal.cKDTree.

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
    used to store stats of the iteration so they will be accessible after the
    iteration is finished.

    Attributes
    ---------------

    max_displacement: np.ndarray
        stores the maximal displacement of every iteration step.

    min_displacement: np.ndarray
        stores the minimal displacement of every iteration step.

    avg_displacement: np.ndarray
        stores the average displacement of every iteration step.

    niter: int
        number of iterations performed.


    Methods
    -------------

    self.add_stat:
        add a new statistic for iteration ``iteration``.
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
        iteration: int,
        min_displacement: float,
        max_displacement: float,
        avg_displacement: float,
    ):
        r"""
        Store new stats at the appropriate place.

        Parameters
        ---------------

        iteration: int
            iteration number

        max_displacement: float
           maximal displacement of this iteration step.

        min_displacement: float
           minimal displacement of this iteration step.

        avg_displacement: float
           average displacement of this iteration step.

        """

        self.max_displacement[iteration] = max_displacement
        self.min_displacement[iteration] = min_displacement
        self.avg_displacement[iteration] = avg_displacement
        self.niter = iteration
        return


class ParticleGenerator(object):
    r"""
    Main class for generating initial conditions for SPH simulations for
    a given density function ``rho(x, ndim)`` following Arth et al. 2019
    (https://arxiv.org/abs/1907.11250).

    Attributes
    -------------

    self.rhofunc: callable
        The density function that is to be reproduced in the initial conditions.
        It must take two positional arguments:
        
        - ``x``: np.ndarray of 3D particle coordinates (even if your initial 
          conditions have lower dimensionality)
        - ``ndim``: integer, number of dimensions that your simulations is to have

    self.boxsize: unyt.unyt_array
        The box size of the simulation.

    self.nx: int
        how many particles along every dimension you want your simulation
        to contain

    self.ndim: int
        how many dimensions you want your simulation to have

    self.unitsys: unyt.unit_systems.UnitSystem
        a unit system that contains the units you want your ICs to have

    self.periodic: bool
        whether the simulation box is periodic or not

    self.kernel: str {'cubic spline',}
        which kernel to use

    self.eta: float, optional
        resolution eta, which defines the number of neighbours used
        independently of dimensionality

    self.rho_max: float or None
        The maximal density within the simulation box. If ``None``, an 
        approximate value will be determined if the rejection sampling to obtain
        an initial particle configuration guess is used.

    self.runparams: RunParams
        A ``RunParams`` instance that contains runtime parameters for the generator.
        The user is encouraged to tinker with these parameters before the iteration
        is commenced.

    self.iterparams: IterData
        An ``IterData`` instance that holds data relevant for iteration steps. The
        user is *strongly discouraged* of fiddling with this.

    self.stats: IterStats
        An ``IterStats`` instance that holds iteration statistics that should be
        accessible after the iteration finished.

    self.npart: int
        total number of particles in simulation

    self.coordinates: unyt.unyt_array
        unyt_array containing the final particle coordinates with shape (self.npart, 3).

    self.masses: unyt.unyt_array
        unyt_array containing the final particle masses with shape (self.npart).

    self.smoothing_lengths: unyt.unyt_array
        unyt_array containing the final particle smoothing lengths with shape (self.npart, 3).

    self.densities: unyt.unyt_array
        unyt_array containing the final particle densities with shape (self.npart).

    self.x: np.ndarray
        unitless particle coordinates that will be worked with/on with shape (self.npart, 3).

    self.m: np.ndarray
        unitless particle masses that will be worked with/on with shape (self.npart).




    Notes
    -----------
    
    + The default workflow should be something like
        ``pg = ParticleGenerator(...)``

        ``pg.runparams.iter_max = 10000``

        ``pg.runparams.whatever_parameter = ...``

        ``pg.initial_setup(...)``

        ``pg.run_iteration()``

        ``print("profit")``
    """

    # simulation parameters
    rhofunc: callable
    boxsize: unyt.unyt_array
    unitsys: unyt.unit_systems.UnitSystem
    nx: int
    ndim: int
    periodic: bool
    kernel: str
    eta: float
    rho_max: Union[float, None]

    runparams: RunParams
    iterparams: IterData
    stats: IterStats

    # derived variables
    npart: int

    # internal checks
    _set_up = False
    _restart_finished = False
    _restarting = False

    # unyt/result arrays
    coordinates: unyt.unyt_array
    masses: unyt.unyt_array
    smoothing_lengths: unyt.unyt_array
    densities: unyt.unyt_array

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
        kernel: str = "cubic spline",
        eta: float = 1.2348,
        rho_max: Union[float, None] = None,
    ):
        r"""

        Parameters
        ---------------

        rho: callable
            The density function that is to be reproduced in the initial conditions.
            It must take two positional arguments:
            
            - ``x``: np.ndarray of 3D particle coordinates (even if your initial 
              conditions have lower dimensionality)
            - ``ndim``: integer, number of dimensions that your simulations is to have

        boxsize: unyt.unyt_array
            The box size of the simulation.

        nx: int
            how many particles along every dimension you want your simulation
            to contain

        ndim: int
            how many dimensions you want your simulation to have

        unitsys: unyt.unit_systems.UnitSystem
            a unit system that contains the units you want your ICs to have

        periodic: bool
            whether the simulation box is periodic or not

        kernel: str {'cubic spline',}
            which kernel to use

        eta: float, optional
            resolution eta, which defines the number of neighbours used
            independently of dimensionality

        rho_max: float or None
            The maximal density within the simulation box. If ``None``, an 
            approximate value will be determined if the rejection sampling to obtain
            an initial particle configuration guess is used.

        """

        if not isinstance(boxsize, unyt.unyt_array):
            raise TypeError("boxsize needs to be a unyt array.")

        if not isinstance(unitsys, unyt.unit_systems.UnitSystem):
            raise TypeError("unitsys needs to be a unyt UnitSystem.")

        if not isinstance(nx, (int, np.integer)):
            raise TypeError("nx needs to be an integer")

        if not isinstance(ndim, (int, np.integer)):
            raise TypeError("ndim needs to be an integer")

        self.rhofunc = rho
        self.boxsize = boxsize
        self.unitsys = unitsys
        self.nx = nx
        self.ndim = ndim
        self.periodic = periodic
        self.kernel = kernel
        self.eta = eta
        self.rho_max = rho_max

        # get some derived quantities
        self.npart = self.nx ** self.ndim
        self.boxsize_to_use = boxsize.to(unitsys["length"]).value
        self.runparams = RunParams()
        self.iterparams = IterData()
        # set up self.stats later tho

        return

    def initial_setup(
        self,
        method: str = "rejection",
        x: Union[unyt.unyt_array, None] = None,
        m: Union[unyt.unyt_array, None] = None,
        max_displ: float = 0.4,
    ):
        r"""
        Run an initial setup for the generator. Allocate arrays, prepare paremeter,
        check whether the given parameters make sense.
        Must be called AFTER the ``self.runparams`` paramters have been tweaked by
        the user.

        Parameters
        -------------

        method: str {"rejection", "uniform", "displaced"}
            If no ``x`` is given, create an initial guess for particle coordinates.
            ``method`` defines which method will be used:

            - ``"rejection"``: Use rejection sampling of the model density function

            - ``"uniform"``: Start off with a uniformly distributed particle configuration

            - ``"displaced"``: Displace particles from an initially uniform distribution
              randomly up to a distance ``max_displ * particle distance along axis`` from
              their original position on the grid

        x: unyt.unyt_array or None, optional
            Initial guess for coordinates of particles. If ``None``, the initial
            guess will be generated by rejection sampling the model density function
            ``self.rhofunc``

        m: unyt.unyt_array or None, optional
            ``unyt.unyt_array`` of particle masses. If ``None``, an array will be created
            such that the total mass in the simulation box given the analytical
            density is reproduced, and all particles will have equal masses.

        max_displ: float, optional
            maximal displacement of a particle initially on an uniform grid along 
            any axis, in units of particle distance along that axis. Is only used
            if ``method = 'displaced'``



        Notes
        -------------------

            + Must be called AFTER the ``self.runparams`` paramters have been tweaked by
        the user.

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

        # if you are restarting, and did this already, don't do it again.
        # this way, the user still can change parameters however they want
        # and call self.initial_setup again.
        if not self._restart_finished:

            # generate first positions if necessary
            if x is None:
                if method == "rejection":
                    self.coordinates = self.rejection_sample_coords()
                elif method == "displaced":
                    self.coordinates = self.generate_displaced_uniform_coords(
                        max_displ=max_displ
                    )
                elif method == "uniform":
                    self.coordinates = self.generate_uniform_coords()
                else:
                    raise ValueError("Unknown coordinate generation method:", method)
            else:
                if not isinstance(x, unyt.unyt_array):
                    raise TypeError("x must be an unyt_array")
                self.coordinates = x.to(self.unitsys["length"])

            #  generate masses if necessary
            if m is None:
                nc = int(10000 ** (1.0 / ndim) + 0.5)  # always use ~ 10000 mesh points
                dx = boxsize / nc

                #  integrate total mass in box
                xc = self.generate_uniform_coords(nx=nc)
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
                if not isinstance(m, unyt.unyt_array):
                    raise TypeError("m must be an unyt_array")
                self.masses = m.to(self.unitsys["mass"])

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
        if not self._restarting:
            if runparams.delta_init is None:
                ip.compute_delta_norm = True
                ip.delta_r_norm = mid
            else:
                # deleta_init is set = -1 during restart.
                # > 0 (or None) means user changed it.
                if runparams.delta_init > 0:
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

    def generate_uniform_coords(
        self, nx: Union[None, int] = None, ndim: Union[None, int] = None
    ):
        r"""
        Generate coordinates for a uniform particle distribution.

        Parameters
        ------------------

        nx: int or None, optional
            How many particles to generate in every dimension. If ``None``, use
            ``self.nx`` grid points.

        ndim: int or None, optional
            How many dimensions to use. If ``None``, use ``self.ndim`` dimensions.

        Returns
        ------------------

        x: unyt.unyt_array 
            unyt.unyt_array of paricle coordinates with shape (``nx**ndim``, 3)
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

    def generate_displaced_uniform_coords(self, max_displ: float = 0.4):
        """
        Get the coordinates for a randomly perturbed uniform particle distribution.
        The perturbation won't exceed ``max_displ`` times the interparticle distance
        along an axis.


        Parameters
        ------------------

        max_displ: float, optional
            maximal displacement of a particle initially on an uniform grid along 
            any axis, in units of particle distance along that axis.

        Returns
        ------------------

        x: unyt.unyt_array 
            unyt.unyt_array of particle coordinates with shape (self.npart, 3)
        """

        nx = self.nx
        boxsize = self.boxsize_to_use
        ndim = self.ndim
        periodic = self.periodic
        npart = nx ** ndim

        # get maximal displacement from uniform grid of any particle along an axis
        maxdelta = max_displ * boxsize / nx

        # generate uniform grid (including units) first
        x = self.generate_uniform_coords()

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

    def rejection_sample_coords(self):
        r"""
        Generate an initial guess for particle coordinates by rejection sampling the
        model density function ``self.rhofunc``.

        Returns
        ------------------

        x: unyt.unyt_array 
            unyt.unyt_array of particle coordinates with shape (self.npart, 3)
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
            xc = self.generate_uniform_coords(nx=nc)
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
        Compute actual smoothing lengths and particle densities.

        Parameters
        -----------------

        nngb: integer or None, optional
            number of neighbours to search for for each particle. If ``None``,
            ``self.iterparams.Nngb_int``  will be used, which is
            the automatically set up value for given parameters during
            initialisation.

        boxsize: np.ndarray or None, optional
            boxsize to be used to generate the tree. If ``None``,
            ``self.iterparams.boxsize_for_tree`` will be used, which is
            the automatically set up value for given parameters during
            initialisation.


        Returns
        --------------

        h: np.ndarray
            numpy array of particle smoothing lengths with shape (``self.npart``)

        rho: np.ndarray
            numpy array of particle densities with shape (``self.npart``)

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
            # tree.query returns index nparts+1 if not enough neighbours were found
            mask = neighs < self.npart
            dist = dist[mask]
            neighs = neighs[mask]
            if neighs.shape[0] == 0:
                raise RuntimeError("Found no neighbour for a particle.")
            h[p] = dist[-1] / kernel_gamma
            for i, n in enumerate(neighs):
                W = kernel_func(dist[i], dist[-1])
                rho[p] += W * self.m[n]

        return h, rho

    def iteration_step(
        self, iteration: int,
    ):
        """
        Perform one step of the actual iteration.

        Parameters
        -------------

        iteration: int
            current iteration number


        Returns
        --------------

        converged: bool
            Whether the iteration satisfies convergence criteria.

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
        redistribute = runparams.redist_freq > 0
        if redistribute:
            redistribute = iteration % runparams.redist_freq == 0
            redistribute = redistribute and runparams.redist_stop >= iteration

        if dump_now or redistribute:

            # first build new tree, get smoothing lengths and densities
            h, rho = self.compute_h_and_rho()

            if dump_now:
                self.dump_current_state(iteration, h, rho)
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
        vol = 1.0
        for d in range(ndim):
            vol *= boxsize[d]

        if ndim == 1:
            hmodel = 0.5 * ipars.Nngb * oneoverrho / oneoverrhosum * vol
        elif ndim == 2:
            hmodel = np.sqrt(ipars.Nngb / np.pi * oneoverrho / oneoverrhosum * vol)
        elif ndim == 3:
            hmodel = np.cbrt(
                ipars.Nngb * 3 / 4 / np.pi * oneoverrho / oneoverrhosum * vol
            )

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
            if neighs.shape[0] == 0:
                raise RuntimeError("Found no neighbour for a particle.")
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
            iteration - 1, min_displacement, max_displacement, avg_displacement
        )

        return converged

    def run_iteration(self):
        """
        Run the actual iteration algorithm to generate initial conditions.
        """
        # this import is only temporary to avoid circular imports.
        # it will be removed before the merge.

        if not self._set_up:
            self.initial_setup()

        # start iteration loop
        iteration = 0

        while iteration < self.runparams.iter_max:

            iteration += 1
            converged = self.iteration_step(iteration)

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

        # if you're dumping intermediate outputs, dump the last one for restarts:
        if self.runparams.dumpfreq > 0:
            self.dump_current_state(iteration, h, rho)

        return

    def redistribute_particles(
        self, h: np.ndarray, rho: np.ndarray, rhoA: np.ndarray,
    ):
        """
        Displace overdense particles into the proximity of underdense particles.


        Parameters
        -----------------

        h: np.ndarray
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

        max_attempts_over = 10 * to_move  # only try your luck, don't force it
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
                while attempts_under < nunder:  # only try your luck, don't force it

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

    def restart(self, filename: str):
        r"""
        Load up settings and particle properties from an intermediate
        dump created with self.dump_current_state().

        Parameters
        ----------------

        filename: str
            filename of the dump to restart from.
        """

        self._restarting = True

        # First load up old arguments

        dumpfile = h5py.File(filename, "r")
        partgen = dumpfile["ParticleGenerator"]

        boxsize = unyt.unyt_array(
            partgen.attrs["boxsize"], partgen.attrs["boxsize_units"]
        )
        nx = partgen.attrs["nx"].astype(int)
        ndim = partgen.attrs["ndim"]
        usys = unyt.unit_systems.UnitSystem(
            "ParticleGenerator UnitSystem",
            partgen.attrs["unit_l"],
            partgen.attrs["unit_m"],
            partgen.attrs["unit_t"],
        )
        periodic = partgen.attrs["periodic"]
        kernel = partgen.attrs["kernel"]
        eta = partgen.attrs["eta"]

        # re-init yourself
        self.__init__(
            self.rhofunc,
            boxsize,
            usys,
            nx,
            ndim,
            periodic=periodic,
            kernel=kernel,
            eta=eta,
        )

        # now set up self.runparams
        self.runparams.iter_max = partgen.attrs["iter_max"]
        self.runparams.iter_min = partgen.attrs["iter_min"]
        self.runparams.converge_thresh = partgen.attrs["converge_thresh"]

        self.runparams.tolerance_part = partgen.attrs["tolerance_part"]
        self.runparams.tolerance_part = partgen.attrs["displ_thresh"]

        self.runparams.delta_reduct = partgen.attrs["delta_reduct"]
        self.runparams.delta_min = partgen.attrs["delta_min"]
        self.runparams.redist_freq = partgen.attrs["redist_freq"]
        self.runparams.redist_frac = partgen.attrs["redist_frac"]
        self.runparams.redist_reduct = partgen.attrs["redist_reduct"]
        self.runparams.dumpfreq = partgen.attrs["dumpfreq"]

        dnorm = partgen.attrs["delta_r_norm"]
        dnorm_min = partgen.attrs["delta_r_norm_min"]
        dumpfile.close()

        # read in particle positions and masses
        dump = load(filename)
        x = dump.gas.coordinates
        m = dump.gas.masses

        # call initial_setup with given x and m
        self.initial_setup(x=x, m=m)

        # overwrite iterparams
        self.iterparams.delta_r_norm = dnorm
        self.iterparams.delta_r_norm_min = dnorm_min

        # make sure that you don't accidentally overwrite delta_r_norm
        # if self.initial_setup is called again
        self.runparams.delta_init = -1
        self.iterparams.compute_delta_norm = False

        # mark that we're finished with the restart
        self._restart_finished = True
        self._restarting = False

        return

    def dump_current_state(
        self, iteration: int, h: np.ndarray, rho: np.ndarray,
    ):
        r"""
        Write an intermediate output of current particle positions, densities,
        masses, and smoothing lengths.
        Use the default swift IC format for the dump, so invent meaningless
        values for required quantities ``velocity`` and ``internal_energy``.

        Parameters
        ----------------

        iteration: int
            current iteration number

        h: np.ndarray
            particle smoothing lengths

        rho: np.ndarray
            particle densities

        """
        ndim = self.ndim
        npart = self.npart
        x = self.x
        m = self.m

        u_l = self.unitsys["length"]
        u_m = self.unitsys["mass"]
        u_t = self.unitsys["time"]

        w = Writer(self.unitsys, self.boxsize)
        w.gas.coordinates = unyt.unyt_array(x, u_l)
        w.gas.smoothing_length = unyt.unyt_array(h, u_l)
        w.gas.masses = unyt.unyt_array(m, u_m)
        w.gas.densities = unyt.unyt_array(rho, u_m / u_l ** ndim)

        # invent some junk to fill up necessary arrays
        w.gas.internal_energy = unyt.unyt_array(
            np.ones(npart, dtype=np.float), u_l ** 2 / u_t ** 2
        )
        w.gas.velocities = unyt.unyt_array(np.zeros(npart, dtype=np.float), u_l / u_t)

        fname = self.runparams.dump_basename + str(iteration).zfill(5) + ".hdf5"
        w.write(fname)

        # Now add extra particle generator data to enable restart
        f = h5py.File(fname, "r+")
        pg = f.create_group("ParticleGenerator")
        pg.attrs["boxsize"] = self.boxsize.value
        pg.attrs["boxsize_units"] = str(self.boxsize.units)
        pg.attrs["nx"] = self.nx
        pg.attrs["ndim"] = self.ndim
        pg.attrs["unit_l"] = str(self.unitsys["length"])
        pg.attrs["unit_m"] = str(self.unitsys["mass"])
        pg.attrs["unit_t"] = str(self.unitsys["time"])
        pg.attrs["periodic"] = self.periodic
        pg.attrs["kernel"] = self.kernel
        pg.attrs["eta"] = self.eta

        pg.attrs["iter_max"] = self.runparams.iter_max
        pg.attrs["iter_min"] = self.runparams.iter_min
        pg.attrs["converge_thresh"] = self.runparams.converge_thresh
        pg.attrs["tolerance_part"] = self.runparams.tolerance_part
        pg.attrs["displ_thresh"] = self.runparams.displ_thresh
        pg.attrs["delta_reduct"] = self.runparams.delta_reduct
        pg.attrs["delta_min"] = self.runparams.delta_min
        pg.attrs["redist_freq"] = self.runparams.redist_freq
        pg.attrs["redist_frac"] = self.runparams.redist_frac
        pg.attrs["redist_reduct"] = self.runparams.redist_reduct
        pg.attrs["dumpfreq"] = self.runparams.dumpfreq
        pg.attrs["dump_basename"] = self.runparams.dump_basename

        # store IterData attributes
        pg.attrs["delta_r_norm"] = self.iterparams.delta_r_norm
        pg.attrs["delta_r_norm_min"] = self.iterparams.delta_r_norm_min

        # add gas adabatic index so when you read the file in in self.restart()
        # you don't get a warning. Note: This will not affect the initial
        # conditions in any way.
        f.create_group("HydroScheme")
        f["HydroScheme"].attrs["Adiabatic index"] = 5.0 / 3

        f.close()

        return
