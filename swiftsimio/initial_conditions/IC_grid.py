#!/usr/bin/env python3


#----------------------------------------
# Grid related functions and classes
#----------------------------------------


import numpy as np
import copy


from particle_hydro_kernel import get_kernel_data

min_neighbour_fact_for_cells = 2 # require minimum of min_neighbour_fact_for_cells * nngb particles in cells for h iteration
ITER_MAX = 100 # max number of iterations to do
EPSILON = 1e-3 # convergence tolerance






def compute_smoothing_lengths(x, m, eta, kernel='cubic spline', ndim=2, periodic=True, ncells = None):
    """
    Compute the smoothing length and density of particles with given positions x and masses m
        x:      particle coordinates
        m:      particle masses
        eta:    "resolution", that defines number of neighbours
        kernel: which kernel to use
        ndim:   number of dimensions
        periodic: whether the gig is periodic
        ncells: a first guess for how many cells in every dimension you want

    returns:
        h:      1D numpy array containing particle smoothing lengths
        rho:    1D numpy array containing particle densities computed with computed smoothing lengths
        grid:   list of cells constituting the grid
        ncells: number of cells in each dimension
        neighbours: list of arrays containing neighbour particle indices for each particle
    """

    grid, ncells = build_grid(x, m, eta, kernel=kernel, ndim=ndim, periodic=periodic, verbose=False, ncells=ncells)

    nparttot = m.shape[0]

    h = np.zeros(nparttot, dtype=np.float)
    rho = np.zeros(nparttot, dtype=np.float)
    neighbours = [[] for i in range(nparttot)]

    kernel_func, kernel_derivative, kernel_gamma = get_kernel_data(kernel, ndim)

    if ndim == 1:
        nngb = 2 * eta * kernel_gamma
    elif ndim == 2:
        nngb = np.pi * (eta * kernel_gamma)**2

    HMAX = 0.5
    HMIN = nparttot**(-1./ndim) * 2 # take at least 2 interparticle distances

    # get a rounded up integer to use in arrays
    nngb = int(nngb + 0.5)
    

    for c in grid:
        nbrs = c.get_neighbours(ncells, ndim, periodic=periodic)
        # get all particles in this neighbourhood
        allparts = []
        for n in nbrs:
            allparts += grid[n].parts

        # get all neighbour particle positions and masses 
        xn = x[allparts]
        mn = m[allparts]

        # now loop over every particle in this cell
        for p in c.parts:
            if ndim == 1:
                r = xn - x[p]
                if (periodic):
                    r[r>0.5] -= 1
                    r[r<-0.5] += 1
                r = np.abs(r)
            elif ndim == 2:
                dx = xn[:,0] - x[p,0]
                if (periodic):
                    dx[dx>0.5] -= 1
                    dx[dx<-0.5] += 1
                dy = xn[:,1] - x[p,1]
                if (periodic):
                    dy[dy>0.5] -= 1
                    dy[dy<-0.5] += 1
                r = np.sqrt(dx**2 + dy**2)
    
            sortind = np.argsort(r)
            r = r[sortind]
            msort = mn[sortind]
            nbrsort = np.array(allparts)[sortind]

            # prepare iterations
            niter = 0
            Hi = r[nngb] # initial guess
            Hiprev = 0. # store previous one

            # start iterating
            while True:
                niter += 1

                i = 0
                while r[i] <= Hi: # find where to stop
                    i += 1
                    if i == len(r): 
                        break

                W = kernel_func(r[:i], Hi)
                dWdr = kernel_derivative(r[:i], Hi)
                ni = W.sum()
                dfdh_sum = (ndim * W + r[:i] * dWdr).sum()

                # compute f and df/dh
                hi = Hi / kernel_gamma
                f = hi**ndim * ni - eta**ndim
                dfdh = hi**(ndim - 1) *(ndim * ni - dfdh_sum)

                Hinew = Hi - f / dfdh
                # exception handling...
                if Hinew > HMAX:
                    Hinew = HMAX
                if Hinew < HMIN:
                    Hinew = HMIN

                if abs(Hinew - Hi) < EPSILON * Hi:
                    h[p] = Hinew / kernel_gamma
                    neighbours[p] = copy.copy(nbrsort[1:i]) # index 0 is particle itself, skip that
                    break
                else:
                    Hiprev = Hi
                    Hi = Hinew

                if niter == ITER_MAX: 
                    Hi = (Hi + Hiprev) * 0.5
                    if Hi < HMAX and Hi > HMIN:
                        break
                    else:
                        print("Hinew:", Hinew, "Hiprev:", Hiprev, "HMAX", HMAX, "HMIN", HMIN, "|", abs(Hinew-Hiprev) ,  EPSILON*Hi)

                    print("Reached max number of iterations for smoothing length. ",
                        "Current convergence rate:", abs(Hinew - Hi)/Hi, 
                        "Hi:", Hi, "Hinew:", Hinew, "dx:", 1./x.shape[0]**(1./ndim)
                        )

                    print("Maybe retry with a higher `min_neighbour_fact_for_cells`?")

                    from matplotlib import pyplot as plt
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    if ndim == 1:
                        # draw grid
                        for i in range(ncells):
                            ax1.plot([(i+1)/ncells, (i+1)/ncells], [0, 1], c='k')
                        # draw smoothing length
                        ax1.errorbar(x=x[p], y=[0.5], xerr=[Hinew], c='red')
                        # draw particles
                        ax1.scatter(x, [0.5 for y in x], s=1, alpha=0.8)
                        # draw problem particle
                        ax1.scatter(x=x[p], y=[0.5], s=4, c='red')
                        

                    elif ndim == 2:
                        # draw grid
                        for i in range(ncells):
                            ax1.plot([(i+1)/ncells, (i+1)/ncells], [0, 1], c='k')
                            ax1.plot([0, 1], [(i+1)/ncells, (i+1)/ncells], c='k')
                        # draw smoothing length
                        ax1.errorbar(x=x[p, 0], y=x[p,1], xerr=[Hinew], yerr=[Hinew], c='red')
                        # draw particles
                        ax1.scatter(x[:,0], x[:, 1], s=1, alpha=0.8)
                        # draw problem particle
                        ax1.scatter(x[p,0], x[p, 1], s=2, c='red')

                    ax1.set_xlim(0, 1)
                    ax1.set_ylim(0, 1)
                    ax1.set_title("Red particle is the problematic one...")

                    plt.show()
                    plt.close()

                    break

            # compute density now
            rhoi = 0.
            i = 0
            while r[i] <= Hi:
                W = kernel_func(r[i], Hi)
                rhoi += msort[i] * W
                i += 1
                if i == len(r): break
            rho[p] = rhoi


    return h, rho, grid, ncells, neighbours











class cell():
    """
    A cell objects to quickly access neighbour lists
    """


    def __init__(self, index):
        self.ind = index
        self.npart = 0
        self.parts = []

        return


    def add_particle(self, partind):
        """
        Add a particle (index) to the list of particles
        of this cell
        """
 
        self.parts.append(partind)
        self.npart += 1
        return


    def get_neighbours(self, ncells, ndim = 2, periodic = True):
        """
        Find all neighbouring cells
        ncells:     number of cells in every dimension
        ndim:       number of dimensions
        periodic:   whether we have periodic boundary conditions
        
        returns:
            list of indices of cells, this cell included
        """

        neighs = [self.ind]

        if ndim == 1:

            left = self.ind - 1
            if left < 0:
                if periodic:
                    neighs.append(left + ncells)

            right = self.ind + 1
            if right > ncells - 1:
                if periodic:
                    neighs.append(right - ncells)



        elif ndim == 2:
            
            add_upper = False
            add_lower = False
            add_left  = False
            add_right = False

            i, j = grid_ij_from_ind(self.ind, ncells)
            
            left = i - 1
            if left >= 0: 
                add_left = True
            else:
                if periodic:
                    left += ncells
                    add_left = True

            right = i + 1
            if right <= ncells - 1: 
                add_right = True
            else:
                if periodic:
                    right -= ncells
                    add_right = True
            
            lower = j - 1
            if lower >= 0: 
                add_lower = True
            else:
                if periodic:
                    lower += ncells
                    add_lower = True

            upper = j + 1
            if upper <= ncells - 1: 
                add_upper = True
            else:
                if periodic:
                    upper -= ncells
                    add_upper = True


            if add_left:
                neighs.append(grid_ind_from_ij(left, j, ncells))
                if add_lower:
                    neighs.append(grid_ind_from_ij(left, lower, ncells))
                if add_upper:
                    neighs.append(grid_ind_from_ij(left, upper, ncells))

            if add_right:
                neighs.append(grid_ind_from_ij(right, j, ncells))
                if add_lower:
                    neighs.append(grid_ind_from_ij(right, lower, ncells))
                if add_upper:
                    neighs.append(grid_ind_from_ij(right, upper, ncells))

            if add_lower:
                neighs.append(grid_ind_from_ij(i, lower, ncells))
            if add_upper:
                neighs.append(grid_ind_from_ij(i, upper, ncells))




        return neighs






def build_grid(x, m, eta, kernel='cubic spline', ndim=2, periodic=True, verbose=True, ncells=None):
    """
    Build a grid and distribute all particles in it.

        x:      particle coordinates
        m:      particle masses
        eta:    "resolution", that defines number of neighbours
        ndim:   number of dimensions
        periodic: whether the gig is periodic
        verbose: write stats to screen when done
        ncells: a first guess for how many cells in every dimension you want

    returns:
        grid:   list of cell objects representing the grid, with
                particles already distributed
        ncells: number of cells in each dimension
    """

    from particle_hydro_kernel import kernel_gamma_1D, kernel_gamma_2D

    nparttot = m.shape[0]


    if ndim == 1:
        # find requested number of neighbours
        nngb = 2 * eta * kernel_gamma_1D[kernel]
        
        # get a first estimate of number of cells
        if ncells is None:
            ncells = int(nparttot / (1.5 *nngb))

    elif ndim == 2:
        # find requested number of neighbours
        nngb = np.pi * (eta * kernel_gamma_2D[kernel])**2

        # get a first estimate of number of cells
        if ncells is None:
            ncells = int(np.sqrt(nparttot / (1.5 *nngb)))


    repeat = True
    it = 0
    while repeat:
        # build grid
        it += 1
        grid = [cell(i) for i in range(ncells**ndim)]

        distribute_particles_on_grid(x, grid, ncells, ndim = ndim)

        # check whether we have enough neighbours everywhere 
        for c in grid:
            nbrs = c.get_neighbours(ncells, ndim, periodic=periodic)
            nbparts = 0
            for n in nbrs:
                nbparts += grid[n].npart

            if nbparts <= min_neighbour_fact_for_cells * nngb:
                ncells -= 1
                print("got too few neighbours in cell", c.ind, 
                        ": particles found:", nbparts, "want:",
                        min_neighbour_fact_for_cells*nngb, 
                        "iteration =", it, "repeating")
                break
        else:
            repeat = False



    if verbose:
        # print some stuff :)
        gridpart = [g.npart for g in grid]
        ngridpart = sum(gridpart)
        ngridmin = min(gridpart)
        ngridmax = max(gridpart)
        ngridmean = ngridpart/ncells**ndim
        print("finished grid building. Ended up with ncells =", ncells)
        print("Grid particle stats:")
        print("    total:", ngridpart, "/", nparttot)
        print("      min:", ngridmin)
        print("      max:", ngridmax)
        print("  average:", ngridmean)


    return grid, ncells











def distribute_particles_on_grid(x, grid, ncells, ndim = 2):
    """
    Distribute particles on the grid's cells
    x :     particle coordinates
    grid:   the grid (list of cells)
    ncells: number of cells in each dimension (integer)
    """

    dx = 1./ ncells

    if ndim == 1:
        for p in range(x.shape[0]):
            i = int(x[p] / dx)
            grid[i].add_particle(p)

    elif ndim == 2:
        for p in range(x.shape[0]):
            i = int(x[p,0] / dx)
            j = int(x[p,1] / dx)
            ind = grid_ind_from_ij(i, j, ncells)
            grid[ind].add_particle(p)

    return







def grid_ij_from_ind(ind, ncells):
    """
    Get i,j grid indices from global cell index
    """

    i = ind % ncells
    j = ind // ncells

    return i, j





def grid_ind_from_ij(i, j, ncells):
    """
    Get global index from i, j cell indices
    """

    return j * ncells + i




