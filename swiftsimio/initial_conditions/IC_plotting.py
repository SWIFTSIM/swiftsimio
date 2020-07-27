import numpy as np
import unyt
from .generate_particles import ic_sim_params


def IC_plot_current_situation(
    save: bool,
    iteration: int,
    x: np.ndarray,
    rho: np.ndarray,
    rho_anal: callable,
    ic_sim_params: ic_sim_params,
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
    
    ic_sim_params: ic_sim_params
        an ``ic_sim_params`` instance containing simulation parameters

    Note
    ---------------

    + For debugging/checking purposes only, not meant to be called.
    """
    from matplotlib import pyplot as plt
    from scipy import stats

    # do this here to avoid circular imports
    from .generate_particles import IC_uniform_coordinates

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
