"""
Generate particle initial conditions that follow some density function.
"""

#--------------------------------------------
# IC generation related stuff
#--------------------------------------------


import numpy as np
from unyt import unyt_array
from math import erf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from .IC_kernel import get_kernel_data

from swiftsimio.optional_packages import KDTree, TREE_AVAILABLE


np.random.seed(666)


def IC_set_IC_params(
            boxsize     = unyt_array([1., 1., 1.], "cm"),
            periodic    = True,
            nx          = 100,
            ndim        = 2,
            unit_l      = None,
            unit_m      = 'g'
        ):
    """
    Change the initial conditions parameters.

    TODO: docs

    returns:
        dict with these parameters.
    """

    icSimParams = {}
    icSimParams['boxsize'] = boxsize
    icSimParams['periodic'] = periodic
    icSimParams['nx'] = nx
    icSimParams['ndim'] = ndim
    if unit_l is None:
        icSimParams['unit_l'] = boxsize.units
        icSimParams['boxsizeToUse'] = boxsize
    else:
        icSimParams['unit_l'] = unit_l
        icSimParams['boxsizeToUse'] = boxsize.to(unit_l)
    icSimParams['unit_m'] = unit_m

    return icSimParams


def IC_set_run_params(
            iter_max                    = 2000,
            convergence_threshold       = 1e-3,
            tolerance_part              = 1e-2,
            displacement_threshold      = 1e-2,
            delta_init                  = 0.1,
            delta_reduction_factor      = 0.98,
            delta_min                   = 1e-4,
            redistribute_frequency      = 20,
            redistribute_fraction       = 0.01,
            no_redistribution_after     = 200,
            plot_at_redistribution      = True, 
            kernel                      = 'cubic spline',
            eta                         = 1.2348
        ):
    """
    Change the global IC generation iteration parameters.

    parameters:

        <int>   iter_max                    max numbers of iterations for generating IC conditions
        <float> convergence_threshold       if enough particles are displaced by distance below 
                                                threshold * mean interparticle distance, stop iterating.
        <float> tolerance_part              tolerance for not converged particle fraction: 
                                                this fraction of particles can be displaced with 
                                                distances > threshold
        <float> displacement_threshold      iteration halt criterion: Don't stop until every particle is 
                                                displaced by distance < threshold * mean interparticle 
                                                distance
        <float> redistribute_frequency      redistribute a handful of particles every 
                                                `redistribute_at_iteration` iteration
        <float> delta_init                  initial normalization constant for particle motion in 
                                                units of mean interparticle distance
        <float> delta_reduction_factor      reduce normalization constant for particle motion by this 
                                                factor after every iteration
        <float> delta_min                   minimal normalization constant for particle motion in units 
                                                of mean interparticle distance
        <float> redistribute_fraction       fraction of particles to redistribute when doing so
        <int>   no_redistribution_after     don't redistribute particles after this iteration
        <bool>  plot_at_redistribution      create and store a plot of the current situation before 
                                                redistributing?

    return:
        dictionnary containing all this data.
    """

    icRunParams = {}

    icRunParams['ITER_MAX'] = iter_max
    icRunParams['CONVERGENCE_THRESHOLD'] = convergence_threshold
    icRunParams['TOLERANCE_PART'] = tolerance_part
    icRunParams['DISPLACEMENT_THRESHOLD'] = displacement_threshold
    icRunParams['DELTA_INIT'] = delta_init
    icRunParams['DELTA_REDUCTION_FACTOR'] = delta_reduction_factor
    icRunParams['DELTA_MIN'] = delta_min
    icRunParams['REDISTRIBUTE_FREQUENCY'] = redistribute_frequency
    icRunParams['REDISTRIBUTE_FRACTION'] = redistribute_fraction
    icRunParams['NO_REDISTRIBUTION_AFTER'] = no_redistribution_after
    icRunParams['PLOT_AT_REDISTRIBUTION'] = plot_at_redistribution
    icRunParams['KERNEL'] = kernel
    icRunParams['ETA'] = eta

    return icRunParams



def IC_uniform_coordinates(icSimParams):
    """
    Get the coordinates for a uniform particle distribution.
    nx:         number of particles in each dimension
    ndim:       number of dimensions

    returns:
        x: np.array((nx**ndim, 3), dtype=float) of coordinates
    """

    nx = icSimParams['nx']
    boxsize = icSimParams['boxsizeToUse']
    ndim = icSimParams['ndim']

    npart = nx**ndim
    x = unyt_array(np.zeros((npart, 3), dtype=np.float), boxsize.units)

    dxhalf = 0.5 * boxsize[0]/nx
    dyhalf = 0.5 * boxsize[1]/nx
    dzhalf = 0.5 * boxsize[2]/nx

    if ndim == 1:
        x[:,0] = np.linspace(dxhalf, boxsize[0]-dxhalf, nx)

    elif ndim == 2:
        xcoords = np.linspace(dxhalf, boxsize[0]-dxhalf, nx)
        ycoords = np.linspace(dyhalf, boxsize[1]-dyhalf, nx)
        for i in range(nx):
            start = i*nx
            stop = (i+1)*nx
            x[start:stop, 0] = xcoords
            x[start:stop, 1] = ycoords[i]

    elif ndim == 3:
        xcoords = np.linspace(dxhalf, boxsize[0]-dxhalf, nx)
        ycoords = np.linspace(dyhalf, boxsize[1]-dyhalf, nx)
        zcoords = np.linspace(dzhalf, boxsize[2]-dzhalf, nx)
        for j in range(nx):
            for i in range(nx):
                start = j*nx**2 + i*nx
                stop = j*nx**2 + (i+1)*nx
                x[start:stop, 0] = xcoords
                x[start:stop, 1] = ycoords[i]
                x[start:stop, 2] = zcoords[j]

    return x





def IC_perturbed_coordinates(icSimParams):
    """
    Get the coordinates for a perturbed uniform particle distribution.
    The perturbation won't exceed the half interparticle distance
    along an axis.

    returns:
        x: np.array((nx**ndim, ndim), dtype=float) of coordinates
    """

    nx = icSimParams['nx']
    boxsize = icSimParams['boxsizeToUse']
    ndim = icSimParams['ndim']
    periodic = icSimParams['periodic']
    npart = nx**ndim 
    maxdelta = 0.4*np.mean(boxsize)/nx # maximal deviation from uniform grid of any particle along an axis

    x = IC_uniform_coordinates(icSimParams) # generate uniform grid first

    for d in range(ndim):
        sign = 2*np.random.randint(0, 2, size=npart) - 1 # get +1 or -1
        amplitude = unyt_array(np.random.uniform(low = 0.0, high = boxsize[d], size=npart)*maxdelta, x.units)
        x[:,d] += sign * amplitude

        if periodic: # correct where necessary
            over = x[:,d] > boxsize[d]
            x[over, d] -= boxsize[d]
            under = x[:,d] < 0.
            x[under] += boxsize[d]
        else:
            # get new random numbers where necessary
            xmax = x[:,d].max()
            xmin = x[:,d].min()
            sign_redo = None
            amplitude_redo = None
            while xmax > boxsize[d] or xmin < 0.:
                
                over = x[:,d] > boxsize[d]
                under = x[:,d] < 0.
                redo = np.logical_or(over, under)

                if sign_red is None:
                    # for first iteration, get arrays in proper shape
                    sign_redo = sign[redo]
                    amplitude_redo = amplitude[redo]

                # first restore previous state
                x[redo,d] -= sign_redo * amplitude_redo

                # then get new guesses, but only where necessary
                nredo = x[redo,d].shape[0]
                sign = 2*np.random.randint(0, 2, size=nredo) - 1 # get +1 or -1
                amplitude = unyt_array(np.random.uniform(low = 0.0, high = boxsize[d], size=nredo)*maxdelta, x.units)
                x[redo,d] += sign_redo * amplitude_redo

                xmax = x[:,d].max()
                xmin = x[:,d].min()

    return x






def IC_sample_coordinates(icSimParams, rho_anal, rho_max = None):
    """
    Randomly sample the density to get initial coordinates    

    parameters:
        nx:         number of particles in each dimension
        rho_anal:   function rho_anal(x, ndim). 
        rho_max:    peak value of the density. If none, an approximate value will be found.
        ndim:       number of dimensions

    returns:
        x: np.array((nx**ndim, ndim), dtype=float) of coordinates
    """

    nx = icSimParams['nx']
    boxsize = icSimParams['boxsizeToUse']
    ndim = icSimParams['ndim']
    periodic = icSimParams['periodic']
    npart = nx**ndim 

    x = np.empty((npart, 3), dtype=np.float)
 
    if rho_max is None:
        # find approximate peak value of rho_max
        nc = 200 # don't cause memory errors with too big of a grid. Also don't worry too much about accuracy.
        xc = IC_uniform_coordinates(IC_set_IC_params(boxsize=boxsize, periodic=periodic, nx=nc, ndim=ndim))
        rho_max = rho_anal(xc.value, ndim).max() * 1.05 # * 1.05: safety measure to make sure you're always above the analytical value

    keep = 0
    coord_threshold = boxsize.value
    while keep < npart:
        
        xr = np.zeros((1, 3), dtype=np.float)
        for d in range(ndim):
            xr[0,d] = np.random.uniform(low=0.0, high=coord_threshold[d])

        if np.random.uniform() <= rho_anal(xr, ndim)/rho_max:
            x[keep] = xr
            keep += 1

    return unyt_array(x, icSimParams['unit_l'])








def generate_IC_for_given_density(rho_anal, icSimParams, icRunParams, x=None, m=None, rho_max = None):
    """
    Generate SPH initial conditions following Arth et al 2019 https://arxiv.org/abs/1907.11250

    rho_anal:   function rho_anal(x, ndim). Should return a numpy array of the analytical function rho(x)
                for given coordinates x, where x is a numpy array.
    nx:         How many particles you want in each dimension
    ndim:       How many dimensions we're working with
    eta:        "resolution", that defines number of neighbours
    icRunParams:   dict containing IC generation parameters as returned from IC_generation_set_params
    x:          Initial guess for coordinates of particles. If none, perturbed uniform initial
                coordinates will be generated.
                Should be numpy array or None.
    m:          Numpy array of particle masses. If None, an array will be created
                such that the total mass in the simulation box given the
                analytical density is reproduced, and all particles will
                have equal masses.
    kernel:     which kernel to use
    periodic:   Whether we're having periodic boundary conditions or not.

    
    returns:
        x:      particle positions
        m:      Numpy array of particle masses.
        rho:    particle densities
        h:      particle smoothing lengths

    """ 

    # safety checks first

    if not TREE_AVAILABLE:
        raise ImportError(
            "The scipy.spatial.cKDTree class is required to search for smoothing lengths."
        )

    try:
        res = rho_anal(np.ones((10,3), dtype='float'), icSimParams['ndim'])
    except TypeError:
        errmsg = "rho_anal must take only a numpy array x of coordinates as an argument."
        raise TypeError(errmsg)
    if type(res) is not np.ndarray:
        raise TypeError("rho_anal needs to return a numpy array as the result.")


    # shortcuts
    nx = icSimParams['nx']
    ndim = icSimParams['ndim']
    periodic = icSimParams['periodic']
    boxsize = icSimParams['boxsizeToUse']
    npart = nx ** ndim
    MID = np.mean(boxsize.value)/nx # mean interparticle distance
    delta_r_norm = icRunParams['DELTA_INIT'] * MID
    delta_r_min = icRunParams['DELTA_MIN'] * MID
    if periodic: #  this sets up whether the tree build is periodic or not
        boxsizeForTree = boxsize.value[:ndim]
    else:
        boxsizeForTree = None
    # kernel data
    kernel_func, _, kernel_gamma = get_kernel_data(icRunParams['KERNEL'], ndim)

    # expected number of neighbours
    if ndim == 1:
        Nngb = 2 * kernel_gamma * icRunParams['ETA']
    elif ndim == 2:
        Nngb = np.pi * (kernel_gamma * icRunParams['ETA'])**2
    elif ndim == 3:
        Nngb = 4 / 3 * np.pi * (kernel_gamma * icRunParams['ETA'])**3

    # round it up for cKDTree
    Nngb_int = int(Nngb + 0.5)


    # generate first positions if necessary
    if x is None:
        x = IC_sample_coordinates(icSimParams, rho_anal, rho_max)

    #  generate masses if necessary
    if m is None:
        nc = 1000 - 250 * ndim # use nc cells for mass integration
        dx = boxsize.value/nc

        #  integrate total mass in box
        newparams = icSimParams.copy()
        newparams['nx'] = nc
        xc = IC_uniform_coordinates(newparams)
        rho_all = rho_anal(xc.value, ndim)
        if rho_all.any() < 0:
            raise ValueError("Found negative densities inside box using the analytical function you provided")
        rhotot = rho_all.sum()
        area = 1.
        for d in range(ndim):
            area *= dx[d]
        mtot = rhotot * area # go from density to mass

        m = np.ones(npart, dtype=np.float) * mtot / npart
        print("Assigning particle mass: {0:.3e}".format(mtot/npart))




    # empty arrays
    delta_r = np.zeros(x.shape, dtype=np.float)
    rho = np.zeros(npart, dtype=np.float)
    h = np.zeros(npart, dtype=np.float)

    # use unitless arrays from this point on
    x_nounit = x.value

    if icRunParams['PLOT_AT_REDISTRIBUTION']:
        # drop a first plot
        tree = KDTree(x_nounit[:,:ndim], boxsize=boxsizeForTree)
        for p in range(npart):
            dist, neighs = tree.query(x_nounit[p, :ndim], k=Nngb_int)
            h[p] = dist[-1] / kernel_gamma
            W = kernel_func(dist, dist[-1])
            rho[p] = (W * m[neighs]).sum()
        IC_plot_current_situation(True, 0, x_nounit, rho, rho_anal, icSimParams)


    # start iteration loop
    iteration = 0

    while True:

        iteration += 1

        # get analytical density at particle positions
        rhoA = rho_anal(x_nounit, ndim)

        # reset displacements
        delta_r.fill(0.)

        # re-distribute particles?
        if iteration % icRunParams['REDISTRIBUTE_FREQUENCY'] == 0:

            # do we need to compute current densities and h's?
            if icRunParams['PLOT_AT_REDISTRIBUTION'] or icRunParams['NO_REDISTRIBUTION_AFTER'] >= iteration:
                # first build new tree
                tree = KDTree(x_nounit[:,:ndim], boxsize=boxsizeForTree)
                for p in range(npart):
                    dist, neighs = tree.query(x_nounit[p, :ndim], k=Nngb_int)
                    h[p] = dist[-1] / kernel_gamma
                    W = kernel_func(dist, dist[-1])
                    rho[p] = (W * m[neighs]).sum()

                #  plot the current situation first?
                if icRunParams['PLOT_AT_REDISTRIBUTION']:
                    IC_plot_current_situation(True, iteration, x_nounit, rho, rho_anal, icSimParams)

                # re-destribute a handful of particles
                if icRunParams['NO_REDISTRIBUTION_AFTER'] >= iteration:
                    x_nounit, touched = redistribute_particles(x_nounit, h, rho, rhoA, iteration, icRunParams, icSimParams)
                    # updated analytical density computations
                    if touched is not None:
                        rhoA[touched] = rho_anal(x_nounit[touched], ndim)


        # compute MODEL smoothing lengths
        oneoverrho = 1./rhoA
        oneoverrhosum = np.sum(oneoverrho)
        
        if ndim == 1:
            hmodel = 0.5 * Nngb * oneoverrho / oneoverrhosum
        elif ndim == 2:
            hmodel = np.sqrt(Nngb / np.pi * oneoverrho / oneoverrhosum)
        elif ndim == 3:
            hmodel = np.cbrt(Nngb * 3 / 4 / np.pi * oneoverrho / oneoverrhosum)

        # build tree, do neighbour loops
        tree = KDTree(x_nounit[:,:ndim], boxsize=boxsizeForTree)

        for p in range(npart):

            dist, neighs = tree.query(x_nounit[p, :ndim], k=Nngb_int)
            # tree.query returns index npart where not enough neighbours are found
            correct = neighs < npart
            dist = dist[correct][1:] # skip first neighbour: that's particle itself
            neighs = neighs[correct][1:]
            dx = x_nounit[p] - x_nounit[neighs]

            if periodic:
                for d in range(ndim):
                    boundary = boxsize[d].value
                    bhalf = 0.5 * boundary
                    dx[dx[:,d] > bhalf, d] -= boundary
                    dx[dx[:,d] < -bhalf, d] += boundary

            for n, Nind in enumerate(neighs): # skip 0: this is particle itself
                hij = (hmodel[p] + hmodel[Nind]) * 0.5
                Wij = kernel_func(dist[n], hij)
                delta_r[p] += hij* Wij / dist[n] * dx[n]

        # finally, displace particles
        delta_r *= delta_r_norm
        x_nounit += delta_r

        # check whether something's out of bounds
        if periodic:
            for d in range(ndim):

                boundary = boxsize[d].value

                xmax = 2*boundary
                while xmax > boundary: 
                    over = x_nounit[:,d] > boundary
                    x_nounit[over, d] -= boundary
                    xmax = x_nounit[:,d].max()

                xmin = -1.
                while xmin < 0.:
                    under = x_nounit[:,d] < 0.
                    x_nounit[under, d] += boundary
                    xmin = x_nounit[:,d].min()
        else:
            # leave it where it was. This is a bit sketchy, better ideas are welcome.
            for d in range(ndim):
                boundary = boxsize[d].value
                x_nounit[x_nounit>boundary] -= delta_r[x_nounit>boundary]
                x_nounit[x_nounit<0.] -= delta_r[x_nounit<0.]

        # reduce delta_r_norm
        delta_r_norm *= icRunParams['DELTA_REDUCTION_FACTOR']
        delta_r_norm = max(delta_r_norm, delta_r_min)

        # get deviation in units of mean interparticle distance
        dev = np.sqrt(delta_r[:,0]**2 + delta_r[:,1]**2 + delta_r[:,2]**2)
        dev /= MID

        max_deviation = dev.max()
        min_deviation = dev.min()
        av_deviation = dev.sum()/dev.shape[0]

        print("Iteration {0:4d}; Displacement [mean interpart dist] Min: {1:8.5f} Average: {2:8.5f}; Max: {3:8.5f};".format(
                iteration, min_deviation, av_deviation, max_deviation))

        if max_deviation < icRunParams['DISPLACEMENT_THRESHOLD']: # don't think about stopping until max < threshold
            unconverged = dev[dev > icRunParams['CONVERGENCE_THRESHOLD']].shape[0]
            if unconverged < icRunParams['TOLERANCE_PART'] * npart:
                print("Convergence criteria are met.")
                break


        if iteration == icRunParams['ITER_MAX']:
            print("Reached max number of iterations without converging. Returning.")
            break

    coords = unyt_array(x_nounit, icSimParams['unit_l'])
    masses = unyt_array(m, icSimParams['unit_m'])

    return coords, masses







def redistribute_particles(x, h, rho, rhoA, iteration, icRunParams, icSimParams):
    """
    Every few steps, manually displace underdense particles into areas of overdense particles

        x:          particle coordinates
        h:          particle smoothing lengths
        rho:        particle densities
        rhoA:       analytical (wanted) density at the particle positions
        iteration:  current iteration of the particle displacement
        icRunParams:   dict containing IC generation parameters as returned from IC_set_run_params()
        icSimParams:   dict containing IC generation parameters as returned from IC_set_IC_params()

    returns:
        x:      particle coordinates
        touched: indices of particles that have been moved around
    """

    # decrease how many particles you move as number of iterations increases
    npart = x.shape[0]
    boxsize = icSimParams['boxsizeToUse']
    ndim = icSimParams['ndim']


    to_move = int(npart * icRunParams['REDISTRIBUTE_FRACTION'] * (1. - (iteration/icRunParams['NO_REDISTRIBUTION_AFTER'])**3))
    to_move = max(to_move, 0.)

    if to_move == 0:
        return x, None


    _, _, kernel_gamma = get_kernel_data(icRunParams['KERNEL'], ndim)

    underdense = rho < rhoA                     # is this underdense particle?
    overdense = rho > rhoA                      # is this overdense particle?
    touched = np.zeros(npart, dtype=np.bool)    # has this particle been touched as target or as to be moved?
    indices = np.arange(npart)                  # particle indices

    moved = 0

    nover = overdense[overdense].shape[0]
    nunder = underdense[underdense].shape[0]
    while moved < to_move:

        # pick an overdense random particle
        oind = indices[overdense][np.random.randint(0, nover)]
        if touched[oind]: continue # skip touched particles

        # do we work with it?
        othresh = (rho[oind] - rhoA[oind])/rho[oind]
        othresh = erf(othresh)
        if np.random.uniform() < othresh:
            
            attempts = 0
            while True:

                attempts += 1
                if attempts == nunder: break # emergency halt
            
                u = np.random.randint(0, nunder)
                uind = indices[underdense][u]

                if touched[uind]: continue # skip touched particles

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
        if icSimParams['periodic']:
            for d in range(ndim):
                x[x[:,d]>boxsize[d], d] -= boxsize[d].value
                x[x[:,d]<0., d] += boxsize[d].value
        else:
            for d in range(ndim):

                # move them away from the edge by a random factor of mean "cell size" boxsize/npart^ndim
                mask = x[:,d] > boxsize[d]
                x[mask, d] = boxsize[d] * (1. - np.random.uniform(x[x>1.0].shape) / npart**(1./ndim))

                mask = x[:,d] < 0.
                x[mask, d] = np.random.uniform(x[mask,d].shape) * boxsize[d]/npart**(1./ndim)

    return x, indices[touched]












def IC_plot_current_situation(save, iteration, x, rho, rho_anal, icSimParams):
    """
    Create a plot of what things look like now. In particular, scatter the
    particle positions and show what the densities currently look like.

    parameters:
        save:       Boolean. Whether to save to file (if True), or just show a plot.
        iteration:  Current iteration number.
        x:          particle positions. Must be numpy array.
        rho:        particle densities. Must be numpy array.
        rho_anal:   analytical expression for the density. Must be of the form
                    rho_anal(x, ndim): return rho; where both x and rho are numpy arrays
        ndim:       How many dimensions we're working with
        """

    boxsize = icSimParams['boxsizeToUse']
    ndim = icSimParams['ndim']

    if x.shape[0] > 5000:
        marker = ','
    else:
        marker = '.'

    if ndim == 1:
        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax1 = fig.add_subplot(111)
        ax1.scatter(x[:,0], rho, s=1, c='k', label="IC")

        r = np.zeros((100, 3), dtype=np.float)
        r[:,0] = np.linspace(0, boxsize[0], 100)
        ax1.plot(r[:,0], rho_anal(r, ndim), label='analytical')
        ax1.set_xlabel("x")
        ax1.set_ylabel("rho")
        ax1.legend()

    elif ndim == 2:
        fig = plt.figure(figsize=(18,6), dpi=200)

        # get rho map
        #  X = np.linspace(0, 1, 20)
        #  Y = np.linspace(0, 1, 20)
        #  X, Y = np.meshgrid(X, Y)
        #  xmap = np.stack((X.ravel(), Y.ravel()), axis=1)
        #  rhomap = rho_anal(xmap).reshape(20,20)
        #
        #  ax1 = fig.add_subplot(1,1,1, projection='3d')
        #  im1 = ax1.plot_surface(X, Y, rhomap, color='red', alpha=0.2)
        #  ax1.scatter(x[:,0], x[:,1], rho, c='blue', depthshade=True, lw=0, s=2)
        #  plt.show()


        ax1 = fig.add_subplot(131, aspect='equal')
        ax2 = fig.add_subplot(132, )
        ax3 = fig.add_subplot(133, )

        # x - y scatterplot

        ax1.scatter(x[:,0], x[:,1], s=1, alpha=0.5, c='k', marker=marker)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # x - rho plot
        rho_mean, edges,_ = stats.binned_statistic(x[:,0], rho, statistic='mean', bins=50)
        rho2mean, edges,_ = stats.binned_statistic(x[:,0], rho**2, statistic='mean', bins=50)
        rho_std = np.sqrt(rho2mean - rho_mean**2)

        r = 0.5 * (edges[:-1] + edges[1:])
        ax2.errorbar(r, rho_mean, yerr=rho_std, label='average all ptcls', lw=1)

        xa = np.linspace(0, 1, 100)
        ya = np.ones(xa.shape) * 0.5
        XA = np.vstack((xa, ya)).T
        ax2.plot(xa, rho_anal(XA, ndim), label='analytical, y = 0.5')

        ax2.scatter(x[:,0], rho, s=1, alpha=0.4, c='k', label='all particles', marker=',')
        selection = np.logical_and(x[:,1] > 0.45, x[:,1] < 0.55)
        ax2.scatter(x[selection,0], rho[selection], s=2, alpha=0.8, c='r', label='0.45 < y < 0.55')

        ax2.legend()
        ax2.set_xlabel("x")
        ax2.set_ylabel("rho")

        # y - rho plot
        rho_mean, edges,_ = stats.binned_statistic(x[:,1], rho, statistic='mean', bins=50)
        rho2mean, edges,_ = stats.binned_statistic(x[:,1], rho**2, statistic='mean', bins=50)
        rho_std = np.sqrt(rho2mean - rho_mean**2)

        r = 0.5 * (edges[:-1] + edges[1:])
        ax3.errorbar(r, rho_mean, yerr=rho_std, label='average all ptcls', lw=1)

        ya = np.linspace(0, 1, 100)
        xa = np.ones(ya.shape) * 0.5
        XA = np.vstack((xa, ya)).T
        ax3.plot(ya, rho_anal(XA, ndim), label='analytical x = 0.5')

        ax3.scatter(x[:,1], rho, s=1, alpha=0.4, c='k', label='all particles', marker=',')
        selection = np.logical_and(x[:,0] > 0.45, x[:,0] < 0.55)
        ax3.scatter(x[selection,1], rho[selection], s=2, alpha=0.8, c='r', label='0.45 < y < 0.55')

        ax3.legend()
        ax3.set_xlabel("y")
        ax3.set_ylabel("rho")


    ax1.set_title("Iteration ="+str(iteration))
    if save:
        plt.savefig('plot-IC-generation-iteration-'+str(iteration).zfill(5)+'.png')
        print("plotted current situation and saved figure to file.")
    else:
        plt.show()
    plt.close()
    
    return
