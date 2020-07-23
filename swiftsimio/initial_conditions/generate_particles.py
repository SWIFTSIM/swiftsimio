"""
Generate particle initial conditions that follow some density function.
"""

#--------------------------------------------
# IC generation related stuff
#--------------------------------------------


import numpy as np
from math import erf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from .IC_grid import compute_smoothing_lengths, build_grid
from .IC_kernel import get_kernel_data



np.random.seed(666)



def IC_generation_set_params(
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
            plot_at_redistribution      = True
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

    icparams = {}

    icparams['ITER_MAX'] = iter_max
    icparams['CONVERGENCE_THRESHOLD'] = convergence_threshold
    icparams['TOLERANCE_PART'] = tolerance_part
    icparams['DISPLACEMENT_THRESHOLD'] = displacement_threshold
    icparams['DELTA_INIT'] = delta_init
    icparams['DELTA_REDUCTION_FACTOR'] = delta_reduction_factor
    icparams['DELTA_MIN'] = delta_min
    icparams['REDISTRIBUTE_FREQUENCY'] = redistribute_frequency
    icparams['REDISTRIBUTE_FRACTION'] = redistribute_fraction
    icparams['NO_REDISTRIBUTION_AFTER'] = no_redistribution_after
    icparams['PLOT_AT_REDISTRIBUTION'] = plot_at_redistribution

    return icparams



def IC_uniform_coordinates(nx, ndim = 2, periodic = True):
    """
    Get the coordinates for a uniform particle distribution.
    nx:         number of particles in each dimension
    ndim:       number of dimensions
    periodic:   whether we have periodic boundary conditions or not

    returns:
        x: np.array((nx**ndim, ndim), dtype=float) of coordinates
    """

    dxhalf = 0.5/nx

    if ndim == 1:
        x = np.linspace(dxhalf, 1-dxhalf, nx)

    elif ndim == 2:
        xcoords = np.linspace(dxhalf, 1-dxhalf, nx)
        ycoords = np.linspace(dxhalf, 1-dxhalf, nx)
        x,y = np.meshgrid(xcoords, ycoords)
        x = np.stack((x.ravel(), y.ravel()), axis=1)

    return x





def IC_perturbed_coordinates(nx, ndim = 2, periodic = True):
    """
    Get the coordinates for a perturbed uniform particle distribution.
    The perturbation won't exceed the half interparticle distance
    along an axis.

    nx:         number of particles in each dimension
    ndim:       number of dimensions
    periodic:   whether we have periodic boundary conditions or not

    returns:
        x: np.array((nx**ndim, ndim), dtype=float) of coordinates
    """

    x = IC_uniform_coordinates(nx, ndim=ndim, periodic=periodic)
    maxdelta = 0.3/nx
    if ndim == 1:
        for i in range(nx):
            sign = 1 if np.random.random() < 0.5 else -1
            x[i] += sign * np.random.random() * maxdelta
    elif ndim == 2:
        for i in range(nx**2):
            sign = 1 if np.random.random() < 0.5 else -1
            x[i][0] += sign * np.random.random() * maxdelta
            sign = 1 if np.random.random() < 0.5 else -1
            x[i][1] += sign * np.random.random() * maxdelta

    return x






def IC_sample_coordinates(nx, rho_anal, rho_max=None, ndim = 2):
    """
    Randomly sample the density to get initial coordinates    

    parameters:
        nx:         number of particles in each dimension
        rho_anal:   function rho_anal(x, ndim). Should return a float of the analytical function rho(x)
                    for given coordinates x
        rho_max:    peak value of the density. If none, an approximate value will be found.
        ndim:       number of dimensions

    returns:
        x: np.array((nx**ndim, ndim), dtype=float) of coordinates
    """

    print("Sampling particle coordinates.")

    npart = nx**ndim
    x = np.empty((npart, ndim), dtype=np.float)
    
    if rho_max is None:
        nc = max(1000, nx)
        xc = IC_uniform_coordinates(nc, ndim=ndim)
        rho_max = rho_anal(xc).max()

    printstep = int(0.01 * npart / 1000) * 1000
    printstep = max(printstep, 1000)

    keep = 0
    while keep < npart:
        
        xr = np.random.uniform(size=(1, ndim))

        if np.random.uniform() <= rho_anal(xr)/rho_max:
            x[keep] = xr
            keep += 1
            if keep % printstep == 0:
                print("{0:8d} / {1:8d}".format(keep, npart))


    return x








def generate_IC_for_given_density(rho_anal, nx, ndim, eta, icparams=IC_generation_set_params(), x=None, m=None, kernel='cubic spline', periodic=True):
    """
    Generate SPH initial conditions following Arth et al 2019 https://arxiv.org/abs/1907.11250

    rho_anal:   function rho_anal(x). Should return a numpy array of the analytical function rho(x)
                for given coordinates x, where x is a numpy array.
    nx:         How many particles you want in each dimension
    ndim:       How many dimensions we're working with
    eta:        "resolution", that defines number of neighbours
    icparams:   dict containing IC generation parameters as returned from IC_generation_set_params
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

    print("Generating initial conditions")


    # safety checks
    try:
        if ndim == 1:
            res = rho_anal(np.linspace(0,1,10))
        if ndim == 2:
            res = rho_anal(np.ones((10,2), dtype='float'))
    except TypeError:
        print("rho_anal must take only a numpy array x of coordinates as an argument.")
        quit()
    if type(res) is not np.ndarray:
        print("rho_anal needs to return a numpy array as the result.")
        quit()


    # first some preparations
    if x is None:
        # generate positions
        x = IC_sample_coordinates(nx, rho_anal, ndim=ndim)

    npart = x.shape[0]

    if m is None:
        # generate masses
        rhotot = 0.

        nc = max(1000, nx) # use nc cells for mass integration
        dx = 1./nc
        # integrate total mass in box
        xc = IC_uniform_coordinates(nc, ndim=ndim, periodic=periodic)
        rhotot = rho_anal(xc).sum()
        mtot = rhotot * dx**ndim # go from density to mass

        m = np.ones(npart, dtype=np.float) * mtot / npart
        print("Assigning particle mass: {0:.3e}".format(mtot/npart))



    delta_r = np.zeros(x.shape, dtype=np.float)

    kernel_func, kernel_derivative, kernel_gamma = get_kernel_data(kernel, ndim)

    # expected number of neighbours
    if ndim == 1:
        Nngb = 2 * kernel_gamma * eta
    elif ndim == 2:
        Nngb = np.pi * (kernel_gamma * eta)**2

    MID = 1./npart**(1./ndim) # mean interparticle distance
    delta_r_norm = icparams['DELTA_INIT'] * MID
    delta_r_min = icparams['DELTA_MIN'] * MID


    if icparams['PLOT_AT_REDISTRIBUTION']:
        # drop a first plot
        h, rho, _, ncells_proper, _ = compute_smoothing_lengths(x, m, eta, 
                    ndim=ndim, periodic=periodic, ncells=None)
        IC_plot_current_situation(True, 0, x, rho, rho_anal, ndim=ndim)


    # start iteration loop
    iteration = 0
    ncells = None
    ncells_proper = None
    while True:

        iteration += 1

        # get analytical density at particle positions
        rhoA = rho_anal(x)

        # re-distribute particles?
        if iteration % icparams['REDISTRIBUTE_FREQUENCY'] == 0:

            # do we need to compute current densities and h's?
            if icparams['PLOT_AT_REDISTRIBUTION'] or icparams['NO_REDISTRIBUTION_AFTER'] >= iteration:
                h, rho, _, ncells_proper, _ = compute_smoothing_lengths(x, m, eta, 
                            kernel=kernel, ndim=ndim, periodic=periodic, ncells=ncells_proper)

            # plot the current situation first?
            if icparams['PLOT_AT_REDISTRIBUTION']:
                IC_plot_current_situation(True, iteration, x, rho, rho_anal, ndim=ndim)
            
            # re-destribute a handful of particles
            if icparams['NO_REDISTRIBUTION_AFTER'] >= iteration:
                x = redistribute_particles(x, h, rho, rhoA, iteration, icparams=icparams,
                            kernel=kernel, ndim=ndim, periodic=periodic)


        # compute MODEL smoothing lengths
        oneoverrho = 1./rhoA
        oneoverrhosum = np.sum(oneoverrho)
        
        if ndim == 1:
            hmodel = np.abs(0.5 * Nngb * oneoverrho / oneoverrhosum)
        elif ndim == 2:
            hmodel = np.sqrt(Nngb / np.pi * oneoverrho / oneoverrhosum)

        # build grid
        grid, ncells =  build_grid(x, m, eta, kernel=kernel, ndim=ndim, periodic=periodic, verbose=False, ncells=ncells)

        # get list of particle neighbours and distances
        neighbours = [[] for p in range(npart)]
        r = [[] for p in range(npart)]

        for c in grid:
            # get all neighbour CELLS of CELL c
            nbrs = c.get_neighbours(ncells, ndim, periodic=periodic)
            # get all particles in this neighbourhood
            allparts = []
            for n in nbrs:
                allparts += grid[n].parts

            # get all neighbour particle positions and masses 
            xn = x[allparts]

            # now loop over every particle in this cell
            for p in c.parts:
                if ndim == 1:
                    rp = xn - x[p]
                    if (periodic):
                        rp[r>0.5] -= 1
                        rp[r<-0.5] += 1
                    r2 = np.abs(r)
                elif ndim == 2:
                    dx = xn[:,0] - x[p,0]
                    if (periodic):
                        dx[dx>0.5] -= 1
                        dx[dx<-0.5] += 1
                    dy = xn[:,1] - x[p,1]
                    if (periodic):
                        dy[dy>0.5] -= 1
                        dy[dy<-0.5] += 1
                    r2 = dx**2 + dy**2


                for n, N in enumerate(allparts):
                    if N == p: # skip yourself
                        continue
                    if hmodel[p]**2 > r2[n]:
                        neighbours[p].append(N)
                        r[p].append(np.sqrt(r2[n]))
 
        # get the actual displacements and densities
        for p in range(npart):

            dx = x[neighbours[p]] - x[p]
            if periodic:
                dx[dx > 0.5] -= 1.
                dx[dx < -0.5] += 1.

            for n,N in enumerate(neighbours[p]):
                hij = (hmodel[p] + hmodel[N]) * 0.5 
                Wij = kernel_func(r[p][n], hij)
                delta_r[p] += hij* Wij / r[p][n] * dx[n]
                #  rho[p] += Wij * m[N]


        # finally, displace particles
        delta_r *= -delta_r_norm
        x += delta_r

        # check whether something's out of bounds
        if periodic:
            xmax = 2.
            xmin = -1.
            while xmax > 1. or xmin < 0.:
                x[x>1.0] -= 1.0
                x[x<0.] += 1.0
                xmax = x.max()
                xmin = x.min()
        else:
            # leave it where it was
            x[x>1.0] -= delta_r[x>1.0]
            x[x<0.] -= delta_r[x<0.]

        # reduce delta_r_norm
        delta_r_norm *= icparams['DELTA_REDUCTION_FACTOR']
        delta_r_norm = max(delta_r_norm, delta_r_min)


        if ndim == 1:
            dev = np.abs(delta_r)
        elif ndim == 2:
            dev = np.sqrt(delta_r[:,0]**2 + delta_r[:,1]**2)

        dev /= MID # get deviation in units of mean interparticle distance

        max_deviation = dev.max()
        min_deviation = dev.min()
        av_deviation = dev.sum()/dev.shape[0]

        print("Iteration {0:4d}; Displacement [mean interpart dist] Min: {1:8.5f} Average: {2:8.5f}; Max: {3:8.5f};".format(
                iteration, min_deviation, av_deviation, max_deviation))

        if max_deviation < icparams['DISPLACEMENT_THRESHOLD']: # don't think about stopping until max < threshold
            unconverged = dev[dev > icparams['CONVERGENCE_THRESHOLD']].shape[0]
            if unconverged < icparams['TOLERANCE_PART'] * npart:
                print("Convergence criteria are met.")
                break


        if iteration == icparams['ITER_MAX']:
            print("Reached max number of iterations without converging. Returning.")
            break


    print("Happy?")
    h, rho, _, ncells_proper, _ = compute_smoothing_lengths(x, m, eta, 
                kernel=kernel, ndim=ndim, periodic=periodic, ncells=ncells_proper)
    IC_plot_current_situation(False, iteration, x, rho, rho_anal, ndim=ndim)


    return x, m, rho, h







def redistribute_particles(x, h, rho, rhoA, iteration, icparams=IC_generation_set_params(), kernel='cubic spline', ndim = 2, periodic = True):
    """
    Every few steps, manually displace underdense particles into areas of overdense particles

        x:          particle coordinates
        h:          particle smoothing lengths
        rho:        particle densities
        rhoA:       analytical (wanted) density at the particle positions
        iteration:  current iteration of the particle displacement
        icparams:   dict containing IC generation parameters as returned from IC_generation_set_params
        kernel:     which kernel to use
        ndim:       number of dimensions
        periodic:   whether the gig is periodic

    returns:
        h:      1D numpy array containing particle smoothing lengths
        rho:    1D numpy array containing particle densities computed with computed smoothing lengths
        neighbours: list of arrays containing neighbour particle indices for each particle
        x:      particle coordinates
    """

    # decrease how many particles you move as number of iterations increases
    npart = x.shape[0]
    to_move = int(npart * icparams['REDISTRIBUTE_FRACTION'] * (1. - (iteration/icparams['NO_REDISTRIBUTION_AFTER'])**3))
    to_move = max(to_move, 0.)

    if to_move == 0:
        return x


    kernel_func, kernel_derivative, kernel_gamma = get_kernel_data(kernel, ndim)


    underdense = rho < rhoA                     # is this underdense particle?
    overdense = rho > rhoA                      # is this overdense particle?
    touched = np.zeros(npart, dtype=np.bool)    # has this particle been touched as target or as to be moved?
    indices = np.arange(npart)                  # particle indices

    moved = 0

    xover = x[overdense]
    xunder = x[underdense]
    while moved < to_move:

        o = np.random.randint(0, xover.shape[0])

        oind = indices[overdense][o]
        if touched[oind]: continue # skip touched particles

        # pick an overdense random particle
        othresh = (rho[oind] - rhoA[oind])/rho[oind]
        othresh = erf(othresh)
        if np.random.uniform() < othresh:
            
            attempts = 0
            while True:

                attempts += 1
                if attempts == xunder.shape[0]: break # emergency halt
            
                u = np.random.randint(0,xunder.shape[0])
                uind = indices[underdense][u]

                if touched[uind]: continue # skip touched particles

                uthresh = (rhoA[uind] - rho[uind]) / rhoA[uind]

                if np.random.uniform() < uthresh:
                    # we have a match!
                    # compute displacement for overdense particle
                    dx = np.zeros(ndim, dtype=np.float)
                    H = kernel_gamma * h[uind]
                    for i in range(dx.shape[0]):
                        sign = 1 if np.random.random() < 0.5 else -1
                        dx[i] = np.random.uniform() * 0.3 * H * sign

                    x[oind] = x[uind] + dx 
                    touched[oind] = True
                    touched[uind] = True
                    moved += 1
                    break



    # check boundary conditions
    if periodic:
        x[x>1.0] -= 1.0
        x[x<0.] += 1.0
    else:
        # move them away from the edge by a random factor of mean "cell size"
        x[x>1.0] = 1.0 - np.random.uniform(x[x>1.0].shape) * 1./npart**(1./ndim)
        x[x<0.] = 0. + np.random.uniform(x[x<0.].shape) * 1./npart**(1./ndim)



    print("Moved", moved, " / ", to_move, " total:", npart)


    return x












def IC_plot_current_situation(save, iteration, x, rho, rho_anal, ndim=2):
    """
    Create a plot of what things look like now. In particular, scatter the
    particle positions and show what the densities currently look like.

    parameters:
        save:       Boolean. Whether to save to file (if True), or just show a plot.
        iteration:  Current iteration number.
        x:          particle positions. Must be numpy array.
        rho:        particle densities. Must be numpy array.
        rho_anal:   analytical expression for the density. Must be of the form
                    rho_anal(x): return rho; where both x and rho are numpy arrays
        ndim:       How many dimensions we're working with
        """

    if x.shape[0] > 5000:
        marker = ','
    else:
        marker = '.'

    if ndim == 1:
        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, rho, s=1, c='k', label="IC")

        r = np.linspace(0, 1, 100)
        ax1.plot(r, rho_anal(r), label='analytical')
        ax1.set_xlabel("x")
        ax1.set_ylabel("rho")

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
        ax2.plot(xa, rho_anal(XA), label='analytical, y = 0.5')

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
        ax3.plot(ya, rho_anal(XA), label='analytical x = 0.5')

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



