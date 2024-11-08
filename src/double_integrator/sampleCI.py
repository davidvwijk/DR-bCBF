"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains plotting functions for obtaining the final figures with the various 
forward invariant sets for different size T. Used to generate Figure 2 in manuscript.

"""

from main_sim import Simulation
from alive_progress import alive_bar
import numpy as np
from plotting import Plotter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


class sampleCI:
    def getyss(self, Tb, xmin):
        """
        Runs a simulation with a particular backup horizon, and obtains the "steady state"
        velocity value used to generate and plot the controlled invariant sets computed online.

        """

        # Load environment
        env = Simulation(
            safety_flag=True,
            verbose=True,
            robust=True,
            dw_bool=True,
        )
        # Set initial conditions and T_b
        env.x0 = np.array([xmin, 0])
        env.backupTime = Tb

        (
            x_full,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = env.sim()

        # Find steady-state velocity value
        bool_arr = abs(np.diff(np.where(x_full[1] < 0, np.NaN, x_full[1]))) < 1e-4
        ys = x_full[1, 1:]
        yss = np.median(ys[bool_arr])

        return yss

    def runMain(self, Tb_array, save_fig=False, show_fig=True):
        """
        Computes steady-state velocity values for various backup horizon times.
        Generates Figure 2 in manuscript.

        """
        xmin = -2
        yss_array = [0]
        total_sim_time = 4
        lwp = 1.8
        legend_sz = 18
        lwp_sets = 2
        matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")

        for i in range(len(Tb_array)):
            Tb = Tb_array[i]
            yss = self.getyss(Tb, xmin)
            yss_array.append(yss)
            # Load environment
            env = Simulation(
                safety_flag=True,
                verbose=True,
                robust=True,
                dw_bool=True,
            )

            env.x0 = np.array([xmin, yss])
            env.backupTime = Tb
            env.total_steps = int(total_sim_time / env.del_t) + 2
            (
                x_full,
                _,
                u_des_full,
                u_act_full,
                intervened,
                _,
                _,
            ) = env.sim()

            if i == 0:
                Plotter.plotter(
                    x_full,
                    u_act_full,
                    intervened,
                    u_des_full,
                    env,
                    phase_plot_1=False,
                    phase_plot_2a=False,
                    phase_plot_2b=False,
                    phase_plot_CI=True,
                    control_plot=False,
                    latex_plots=True,
                    save_plots=False,
                    show_plots=False,
                )
                ax = plt.gca()
                lblsize = legend_sz * 1.35
            else:
                # Plot different C_I's
                x1 = x_full[0, :]
                x2 = x_full[1, :]
                booly = x2 >= 0
                plt.plot(
                    x1[booly],
                    x2[booly],
                    color=[216 / 255, 110 / 255, 204 / 255],
                    alpha=1,
                    linewidth=lwp_sets,
                )
                plt.fill_between(
                    x1[booly],
                    0,
                    x2[booly],
                    color=[242 / 255, 207 / 255, 238 / 255],
                    alpha=1,
                )

            if i == len(Tb_array) - 1:
                # Sample point and run
                # Load environment
                env = Simulation(
                    safety_flag=True,
                    verbose=True,
                    robust=True,
                    dw_bool=True,
                )

                env.x0 = np.array([-1.58, 0])
                env.backupTime = Tb
                env.total_steps = int(total_sim_time / env.del_t) + 2
                (
                    x_full,
                    _,
                    u_des_full,
                    u_act_full,
                    intervened,
                    _,
                    _,
                ) = env.sim()
                Plotter.plotter(
                    x_full,
                    u_act_full,
                    intervened,
                    u_des_full,
                    env,
                    phase_plot_1=False,
                    phase_plot_2a=False,
                    phase_plot_2b=False,
                    phase_plot_CI=False,
                    control_plot=False,
                    latex_plots=True,
                    save_plots=False,
                    show_plots=False,
                )
                # Plot trajectory
                label = r"$\boldsymbol{\phi}^{d}(t,\boldsymbol{x}_0)" + "$"

                ax.plot(
                    x_full[0, :],
                    x_full[1, :],
                    "--",
                    color="black",
                    linewidth=lwp,
                    label=label,
                )

                ax.plot(
                    x_full[0, 0],
                    x_full[1, 0],
                    "k*",
                    markersize=9,
                    label=r"$\boldsymbol{x}_0$",
                )

                rta_points = len(env.backupTrajs[0])
                gw_edge_color = "#bababa"

                e_tstep = 3
                if env.backupTrajs:
                    max_numBackup = 16
                    min_numBackup = 2
                    for k, xy in enumerate(env.backupTrajs):
                        if k > min_numBackup and k < max_numBackup:
                            circ = []
                            circF = []
                            for j in np.arange(0, rta_points, e_tstep):
                                r_t = env.delta_array[j]
                                if j == 0:
                                    label = "$\delta_{\\rm max}(\\tau)$"
                                else:
                                    label = None
                                cp = patches.Circle(
                                    (xy[j, 0], xy[j, 1]),
                                    r_t,
                                    color=gw_edge_color,
                                    fill=False,
                                    linestyle="--",
                                    label=label,
                                )
                                if k == min_numBackup + 1:
                                    ax.add_patch(cp)
                                circ.append(cp)
                                if j == rta_points - e_tstep:
                                    cpF = patches.Circle(
                                        (xy[-1, 0], xy[-1, 1]),  # final norm bound
                                        r_t,
                                        color="red",
                                        fill=False,
                                        linestyle="--",
                                    )
                                    if k == 0:
                                        ax.add_patch(cpF)
                                    circF.append(cpF)

                            coll = PatchCollection(
                                circ,
                                zorder=100,
                                facecolors=("none",),
                                edgecolors=(gw_edge_color,),
                                linewidths=(1,),
                                linestyle=("--",),
                            )
                            ax.add_collection(coll)
                            collF = PatchCollection(
                                circF,
                                zorder=100,
                                facecolors=("none",),
                                edgecolors=("red",),
                                linewidths=(1,),
                                linestyle=("--",),
                            )
                            ax.add_collection(collF)
                            if k == min_numBackup + 1:
                                label = r"$\boldsymbol{\phi}_{\rm b}^{n}(\tau,\boldsymbol{x})$"
                            else:
                                label = None
                            ax.plot(
                                xy[:, 0],
                                xy[:, 1],
                                color=gw_edge_color,
                                linewidth=1.7,
                                label=label,
                                zorder=1,
                            )

                ax.legend(
                    fontsize=legend_sz,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.01),
                    fancybox=True,
                    shadow=True,
                    framealpha=1,
                    ncol=2,
                )

            ax.text(
                -1.9,
                (yss_array[i + 1] - yss_array[i]) / 2 + yss_array[i] - 0.04,
                "$\mathcal{C}_{{\\rm I}, T = " + str(Tb) + "}$",
                fontsize=lblsize,
            )

        dpi = 500

        if save_fig:
            for i in plt.get_fignums():
                plt.figure(i)
                plt.savefig("plots/figure%d.png" % i, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()

        return yss


if __name__ == "__main__":
    # Backup time array
    Tb = np.array([0.5, 1, 1.25])

    sCI = sampleCI()
    yss = sCI.runMain(Tb, save_fig=False, show_fig=True)
