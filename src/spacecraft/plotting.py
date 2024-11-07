"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains plotting functions.

"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import math
from PIL import Image


class Plotter:
    def plotter(
        x,
        u_act,
        intervening,
        u_p,
        env,
        omegas_plot=True,
        control_plot=True,
        norm_plot=True,
        norm_u_plot=True,
        sphere_plot=True,
        latex_plots=False,
        save_plots=False,
        show_plots=True,
    ):

        # Define constants and extract values

        title_sz, xaxis_sz, legend_sz, ticks_sz = 20, 21, 21, 16
        title_flag = False
        lwp = 2.4
        x1 = x[0, :]
        x2 = x[1, :]
        x3 = x[2, :]
        delta_t = env.del_t
        t_span = np.arange(u_act.shape[1]) * delta_t
        t_span_u = np.arange(u_act.shape[1] - 1) * delta_t
        bounds = env.u_bounds

        if latex_plots:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                }
            )
            plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

        def plotEllipsoid(
            ax,
            coefs,
            color,
            alpha,
            label,
        ):
            rx, ry, rz = 1 / np.sqrt(coefs)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = rx * np.outer(np.cos(u), np.sin(v))
            y = ry * np.outer(np.sin(u), np.sin(v))
            z = rz * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(
                x,
                y,
                z,
                color=color,
                alpha=alpha,
                linewidth=0,
                label=label,
            )

        def plotSphere(ax, center, radius, color, alpha, label):
            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
            x = radius * np.cos(u) * np.sin(v)
            y = radius * np.sin(u) * np.sin(v)
            z = radius * np.cos(v)
            # color = "#F97306"
            c1 = ax.plot_surface(
                center[0] - x,
                center[1] - y,
                center[2] - z,
                color=color,
                alpha=alpha,
                linewidth=0,
                label=label,
            )
            c1._facecolor2d = c1._facecolor3d
            c1._edgecolor2d = c1._edgecolor3d

        def plotSphere_epsilon(ax, center, radius, color, linewidth, label):
            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
            x = radius * np.cos(u) * np.sin(v)
            y = radius * np.sin(u) * np.sin(v)
            z = radius * np.cos(v)
            # color = "#F97306"
            c1 = ax.plot_wireframe(
                center[0] - x,
                center[1] - y,
                center[2] - z,
                color=color,
                linewidth=linewidth,
                label=label,
            )

        def splitfun(l, n):
            output = []
            t = []
            for i in range(1, len(l)):
                if abs(l[i] - l[i - 1]) < n:
                    t.append(l[i])
                else:
                    output.append(t)
                    t = [l[i]]
            output.append(t)
            return output

        if omegas_plot:
            ax = plt.figure(figsize=(10, 7), dpi=100)
            ax = ax.add_subplot(111)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)

            plt.plot(
                t_span,
                x1,
                color="r",
                linewidth=lwp,
                label="$\omega_1$ ($\\frac{\\rm rad}{\\rm s}$)",
            )
            plt.plot(
                t_span,
                x2,
                color="g",
                linewidth=lwp,
                label="$\omega_2$ ($\\frac{\\rm rad}{\\rm s}$)",
            )
            plt.plot(
                t_span,
                x3,
                color="b",
                linewidth=lwp,
                label="$\omega_3$ ($\\frac{\\rm rad}{\\rm s}$)",
            )

            if intervening:
                lists = splitfun(intervening, 2)

            for k in range(3):
                for j in range(len(lists)):
                    if k == 0 and j == 0:
                        label = "Safety Intervention"
                    else:
                        label = None

                    plt.plot(
                        t_span[lists[j]],
                        x[k, lists[j]],
                        color="magenta",
                        linewidth=lwp * 1.03,
                        label=label,
                    )

            y1_fill = np.ones(x.shape[1]) * 0
            y2_fill = np.ones(x.shape[1]) * env.omega_max

            ax.legend(fontsize=legend_sz, loc="upper right")
            plt.grid(True)
            if save_plots:
                plt.savefig("plots/omegas1.svg", dpi=1000)
                plt.savefig("plots/omegas1.png", dpi=1000)

        if norm_plot:
            ax = plt.figure(figsize=(10, 7), dpi=100)
            ax = ax.add_subplot(111)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)

            plt.axhline(env.omega_max, color="k", linestyle="--")
            y1_fill = np.ones(x.shape[1]) * 0
            y2_fill = np.ones(x.shape[1]) * env.omega_max
            ax.fill_between(
                t_span,
                y1_fill,
                y2_fill,
                color=(240 / 255, 255 / 255, 240 / 255),  # Green, safe set
            )
            ax.fill_between(
                t_span,
                y2_fill,
                2 * env.omega_max * np.ones(x.shape[1]),
                color=(255 / 255, 230 / 255, 230 / 255),  # Red, unsafe set
            )

            if env.backupTrajs:
                norms = []
                for j in range(len(env.backupTrajs)):
                    norm = np.zeros(len(env.backupTrajs[j]))
                    for z in range(len(env.backupTrajs[j])):
                        norm[z] = np.linalg.norm(env.backupTrajs[j][z], 2)
                    norms.append(norm)

                tmax_b = env.backupTime
                # Backup trajectory points
                rtapoints = int(math.floor(tmax_b / env.del_t))
                tspan_backup = np.linspace(0, env.backupTime - env.del_t, rtapoints)
                for i, norm in enumerate(norms):
                    if i == 0:
                        # label = "Nominal Backup Flow"
                        label = r"$\boldsymbol{\phi}^{n}_{\rm b} (\tau, \boldsymbol{\omega})$"
                    else:
                        label = None
                    plt.plot(
                        tspan_backup + env.del_t * i * env.backup_save_N,
                        norm,
                        color="cyan",
                        linewidth=lwp * 0.8,
                        label=label,
                    )

            norms = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                norms[i] = np.linalg.norm(x[:, i], 2)

            plt.plot(
                t_span,
                norms,
                color="blue",
                linewidth=lwp,
                label=None,
            )

            ax.set_xlim([0, t_span[-1]])
            ax.set_ylim([0, env.omega_max * 1.05])
            if intervening:
                lists = splitfun(intervening, 2)
                for i in range(len(lists)):
                    if i == 0:
                        # label = "Safety Intervention"
                        label = None
                    else:
                        label = None
                    ax.plot(
                        t_span[lists[i]],
                        norms[lists[i]],
                        color="magenta",
                        linewidth=lwp * 1.03,
                        label=label,
                    )

            unsafe_i = norms > env.omega_max
            if any(unsafe_i):
                plt.scatter(
                    t_span[unsafe_i],
                    norms[unsafe_i],
                    color="red",
                    marker="x",
                    s=40,
                    label="Violation",
                    zorder=10,
                )

            plt.ylabel(
                r"$\| \boldsymbol{\omega} \|$" + "($\\frac{\\rm rad}{\\rm s}$)",
                fontsize=xaxis_sz,
            )
            plt.xlabel("time (s)", fontsize=xaxis_sz)
            ax.legend(fontsize=legend_sz, loc="lower right")
            plt.grid(True)
            if save_plots:
                plt.savefig("plots/omegas.svg", dpi=1000)
                plt.savefig("plots/omegas.png", dpi=1000)

        plot_backupSpheres = True

        # For sphere plot and used by norm_u
        dwr = 3714 / 4000
        dhr = 3022 / 3200
        p1 = False
        if sphere_plot:
            # ax = plt.figure(figsize=(10, 7), dpi=100)
            if p1:
                ax = plt.figure(figsize=(8, 8), dpi=100)
            else:
                ax = plt.figure(figsize=(10, 8), dpi=100)
            ax = ax.add_subplot(111, projection="3d")
            ax.set_xlabel("$\omega_1$ ($\\frac{\\rm rad}{\\rm s}$)", fontsize=xaxis_sz)
            ax.set_ylabel("$\omega_2$ ($\\frac{\\rm rad}{\\rm s}$)", fontsize=xaxis_sz)
            ax.set_zlabel("$\omega_3$ ($\\frac{\\rm rad}{\\rm s}$)", fontsize=xaxis_sz)
            ax.set_box_aspect((1, 1, 1))
            ax.grid(True)
            ax.tick_params(axis="x", which="major", pad=2.5)
            ax.tick_params(axis="y", which="major", pad=2.5)
            ax.tick_params(axis="z", which="major", pad=7)
            ax.xaxis.set_tick_params(labelsize=ticks_sz)
            ax.yaxis.set_tick_params(labelsize=ticks_sz)
            ax.zaxis.set_tick_params(labelsize=ticks_sz)
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10
            ax.zaxis.labelpad = 12

            plotSphere(
                ax,
                np.array([0, 0, 0]),
                env.omega_max,
                "red",
                0.2,
                # "$\| \omega \|$ Constraint",
                "$\mathcal{C}_{\\rm S} $",
            )

            backup_coeffs = (1 / (2 * env.gamma)) * np.diag(env.J)
            plotEllipsoid(ax, backup_coeffs, "cyan", 0.2, "$\mathcal{C}_{\\rm B} $")
            rtapoints = int(math.floor(env.backupTime / env.del_t))
            max_radius = 0.8 * env.omega_max
            for axis in "xyz":
                getattr(ax, "set_{}lim".format(axis))((-max_radius, max_radius))

            ax.plot(x1, x2, x3, color="blue", linewidth=lwp, label=None)

            if intervening:
                lists = splitfun(intervening, 2)
                for i in range(len(lists)):
                    if i == 0:
                        # label = "Safety Intervention"
                        label = None
                    else:
                        label = None
                    ax.plot(
                        x1[lists[i]],
                        x2[lists[i]],
                        x3[lists[i]],
                        color="magenta",
                        linewidth=lwp * 1.03,
                        label=label,
                    )

            n_etSphere = 3
            if env.backupTrajs:
                for i in np.arange(0, len(env.backupTrajs), step=2):
                    if i == 0:
                        label = r"$\boldsymbol{\phi}^{n}_{\rm b} (\tau, \boldsymbol{\omega})$"
                    else:
                        label = None

                    x_c = env.backupTrajs[i][:, 0]
                    y_c = env.backupTrajs[i][:, 1]
                    z_c = env.backupTrajs[i][:, 2]

                    if (
                        env.robust
                        and plot_backupSpheres
                        and i % n_etSphere == 0
                        and i != 0
                    ):
                        for j in np.arange(0, rtapoints, step=4):
                            plotSphere_epsilon(
                                ax,
                                np.array([x_c[j], y_c[j], z_c[j]]),
                                env.delta_array[j],
                                "gray",
                                0.1,
                                None,
                            )

                    plt.plot(
                        x_c,
                        y_c,
                        z_c,
                        color="cyan",
                        linewidth=lwp * 0.75,
                        label=label,
                    )
            ax.plot(
                x1[0],
                x2[0],
                x3[0],
                "k*",
                markersize=8,
                label=r"$\boldsymbol{\omega_0}$",
            )
            ax.view_init(elev=29, azim=128)
            if p1:
                ax.legend(
                    fontsize=legend_sz,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.06),
                    fancybox=True,
                    shadow=True,
                    ncol=2,
                )
                plt.tight_layout(pad=3)
                if save_plots:
                    plt.savefig("plots/3d_plot.png", dpi=400)
            else:
                ax.legend(
                    fontsize=legend_sz,
                    loc="center left",
                    bbox_to_anchor=(-0.163, 0.5),
                    fancybox=True,
                    shadow=True,
                )
                plt.tight_layout()
                if save_plots:
                    # dpi = 225
                    dpi = 425
                    plt.savefig("plots/3d_plot_2.png", dpi=dpi)
                    im = Image.open("plots/3d_plot_2.png")
                    width, height = im.size
                    im1 = im.crop((0, height - dhr * height, dwr * width, height))
                    im1.save("plots/3d_plot_crop.png")

        if control_plot:
            ax = plt.figure(figsize=(10, 7), dpi=100)
            ax = ax.add_subplot(111)
            for i in range(np.shape(u_p)[0]):
                ax.grid(True)
                plt.xticks(fontsize=ticks_sz)
                plt.yticks(fontsize=ticks_sz)
                if i == 0:
                    color = "#231942"
                elif i == 1:
                    color = "#9F86C0"
                else:
                    color = "#E0B1CB"

                ax.plot(
                    t_span_u,
                    u_p[i, 1:],
                    "--",
                    color=color,
                    # label="$u_{\\rm des," + f"{i+1}" + "}$",
                    label="$u_{\\rm p," + f"{i+1}" + "}$",
                    linewidth=lwp,
                )
                ax.plot(
                    t_span_u,
                    u_act[i, 1:],
                    "-",
                    color=color,
                    # label="$u_{\\rm act," + f"{i+1}" + "}$",
                    label="$u_{\\rm safe," + f"{i+1}" + "}$",
                    linewidth=lwp,
                )
                if i == 0:
                    if title_flag:
                        ax.set_title("Control Inputs vs. Time", fontsize=title_sz)
                    ax.set_ylabel("control torque (Nm)", fontsize=xaxis_sz)

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=3,
                fancybox=True,
                fontsize=legend_sz,
                shadow=True,
            )

            plt.xlabel("time (s)", fontsize=xaxis_sz)
            if save_plots:
                plt.savefig("plots/control_plot.svg", dpi=1000)
                plt.savefig("plots/control_plot.png", dpi=1000)

        if norm_u_plot:

            xaxis_sz = 19
            legend_sz = 17

            fig = plt.figure(figsize=(11.69, 7), dpi=100)
            # fig = plt.figure(figsize=((dw/dh)*7, 7), dpi=100)
            ax = fig.add_subplot(2, 1, 1)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)

            plt.axhline(env.omega_max, color="k", linestyle="--")
            y1_fill = np.ones(x.shape[1]) * 0
            y2_fill = np.ones(x.shape[1]) * env.omega_max
            ax.fill_between(
                t_span,
                y1_fill,
                y2_fill,
                color=(240 / 255, 255 / 255, 240 / 255),  # Green, safe set
            )
            ax.fill_between(
                t_span,
                y2_fill,
                2 * env.omega_max * np.ones(x.shape[1]),
                color=(255 / 255, 230 / 255, 230 / 255),  # Red, unsafe set
            )

            if env.backupTrajs:
                norms = []
                for j in range(len(env.backupTrajs)):
                    norm = np.zeros(len(env.backupTrajs[j]))
                    for z in range(len(env.backupTrajs[j])):
                        norm[z] = np.linalg.norm(env.backupTrajs[j][z], 2)
                    norms.append(norm)

                # Backup trajectory points
                tmax_b = env.backupTime
                rtapoints = int(math.floor(tmax_b / env.del_t))
                tspan_backup = np.linspace(0, env.backupTime - env.del_t, rtapoints)
                for i, norm in enumerate(norms):
                    if i == 0:
                        # label = "Nominal Backup Flow"
                        label = r"$\boldsymbol{\phi}^{n}_{\rm b} (\tau, \boldsymbol{\omega})$"
                    else:
                        label = None
                    plt.plot(
                        tspan_backup + env.del_t * i * env.backup_save_N,
                        norm,
                        color="cyan",
                        linewidth=lwp * 0.8,
                        label=label,
                    )

            norms = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                norms[i] = np.linalg.norm(x[:, i], 2)

            plt.plot(
                t_span,
                norms,
                color="blue",
                linewidth=lwp,
                label=None,
            )

            ax.set_xlim([0, t_span[-1]])
            ax.set_ylim([0, env.omega_max * 1.05])
            if intervening:
                lists = splitfun(intervening, 2)
                for i in range(len(lists)):
                    if i == 0:
                        # label = "Safety Intervention"
                        label = None
                    else:
                        label = None
                    ax.plot(
                        t_span[lists[i]],
                        norms[lists[i]],
                        color="magenta",
                        linewidth=lwp * 1.03,
                        label=label,
                    )

            unsafe_i = norms > env.omega_max
            if any(unsafe_i):
                plt.scatter(
                    t_span[unsafe_i],
                    norms[unsafe_i],
                    color="red",
                    marker="x",
                    s=40,
                    label="Violation",
                    zorder=10,
                )

            plt.ylabel(
                r"$\| \boldsymbol{\omega} \|$" + "($\\frac{\\rm rad}{\\rm s}$)",
                fontsize=xaxis_sz,
            )
            ax.legend(
                fontsize=legend_sz,
                fancybox=True,
                shadow=True,
                loc="lower right",
            )
            plt.grid(True)

            ax = fig.add_subplot(2, 1, 2)
            for i in range(np.shape(u_p)[0]):
                ax.grid(True)
                plt.xticks(fontsize=ticks_sz)
                plt.yticks(fontsize=ticks_sz)
                ax.set_xlim([0, t_span[-1]])  # Adjust the x-axis limits
                ax.set_ylim([1.05 * bounds[i][0], 1.05 * bounds[i][1]])

                if i == 0:
                    color = "#231942"
                elif i == 1:
                    color = "#9F86C0"
                else:
                    color = "#E0B1CB"

                ax.plot(
                    t_span_u,
                    u_p[i, 1:],
                    "--",
                    color=color,
                    # label="$u_{\\rm des," + f"{i+1}" + "}$",
                    label="$u_{\\rm p," + f"{i+1}" + "}$",
                    linewidth=lwp,
                )
                ax.plot(
                    t_span_u,
                    u_act[i, 1:],
                    "-",
                    color=color,
                    # label="$u_{\\rm act," + f"{i+1}" + "}$",
                    label="$u_{\\rm safe," + f"{i+1}" + "}$",
                    linewidth=lwp,
                )
                if i == 0:
                    if title_flag:
                        ax.set_title("Control Inputs vs. Time", fontsize=title_sz)
                    ax.set_ylabel("control torque (Nm)", fontsize=xaxis_sz)

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=3,
                # fancybox=True,
                fontsize=legend_sz,
                # shadow=True,
            )

            plt.xlabel("time (s)", fontsize=xaxis_sz)

            if save_plots:
                dpi = 500  # 250
                plt.savefig("plots/norm_u_plot.png", dpi=dpi, bbox_inches="tight")

        if show_plots:
            plt.show()

    def comparison_plotter(
        x_vanilla,
        x,
        u_act,
        intervening,
        u_p,
        env,
        norm_u_plot=True,
        sphere_plot=True,
        latex_plots=False,
        save_plots=False,
        show_plots=True,
    ):
        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                super().__init__((0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def do_3d_projection(self, renderer=None):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

                return np.min(zs)

        # Define constants and extract values

        title_sz, xaxis_sz, legend_sz, ticks_sz = 20, 21, 21, 16
        title_flag = False
        lwp = 2.4
        x1 = x[0, :]
        x2 = x[1, :]
        x3 = x[2, :]

        x1_v = x_vanilla[0, :]
        x2_v = x_vanilla[1, :]
        x3_v = x_vanilla[2, :]

        delta_t = env.del_t
        t_span = np.arange(u_act.shape[1]) * delta_t
        t_span_u = np.arange(u_act.shape[1] - 1) * delta_t
        bounds = env.u_bounds

        if latex_plots:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                }
            )
            plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

        def plotEllipsoid(
            ax,
            coefs,
            color,
            alpha,
            label,
        ):
            rx, ry, rz = 1 / np.sqrt(coefs)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = rx * np.outer(np.cos(u), np.sin(v))
            y = ry * np.outer(np.sin(u), np.sin(v))
            z = rz * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(
                x,
                y,
                z,
                color=color,
                alpha=alpha,
                linewidth=0,
                label=label,
            )

        def plotSphere(ax, center, radius, color, alpha, label):
            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
            x = radius * np.cos(u) * np.sin(v)
            y = radius * np.sin(u) * np.sin(v)
            z = radius * np.cos(v)
            # color = "#F97306"
            c1 = ax.plot_surface(
                center[0] - x,
                center[1] - y,
                center[2] - z,
                color=color,
                alpha=alpha,
                linewidth=0,
                label=label,
            )
            c1._facecolor2d = c1._facecolor3d
            c1._edgecolor2d = c1._edgecolor3d

        def plotSphere_epsilon(ax, center, radius, color, linewidth, label):
            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
            x = radius * np.cos(u) * np.sin(v)
            y = radius * np.sin(u) * np.sin(v)
            z = radius * np.cos(v)
            # color = "#F97306"
            c1 = ax.plot_wireframe(
                center[0] - x,
                center[1] - y,
                center[2] - z,
                color=color,
                linewidth=linewidth,
                label=label,
            )

        def splitfun(l, n):
            output = []
            t = []
            for i in range(1, len(l)):
                if abs(l[i] - l[i - 1]) < n:
                    t.append(l[i])
                else:
                    output.append(t)
                    t = [l[i]]
            output.append(t)
            return output

        plot_backupSpheres = True

        # For sphere plot and used by norm_u
        dwr = 3726 / 4000
        dhr = 3054 / 3200
        p1 = False
        if sphere_plot:
            # ax = plt.figure(figsize=(10, 7), dpi=100)
            if p1:
                ax = plt.figure(figsize=(8, 8), dpi=100)
            else:
                ax = plt.figure(figsize=(10, 8), dpi=100)
            ax = ax.add_subplot(111, projection="3d")
            ax.set_xlabel("$\omega_1$ ($\\frac{\\rm rad}{\\rm s}$)", fontsize=xaxis_sz)
            ax.set_ylabel("$\omega_2$ ($\\frac{\\rm rad}{\\rm s}$)", fontsize=xaxis_sz)
            ax.set_zlabel("$\omega_3$ ($\\frac{\\rm rad}{\\rm s}$)", fontsize=xaxis_sz)
            ax.set_box_aspect((1, 1, 1))
            ax.grid(True)
            ax.tick_params(axis="x", which="major", pad=2.5)
            ax.tick_params(axis="y", which="major", pad=2.5)
            ax.tick_params(axis="z", which="major", pad=7)
            ax.xaxis.set_tick_params(labelsize=ticks_sz)
            ax.yaxis.set_tick_params(labelsize=ticks_sz)
            ax.zaxis.set_tick_params(labelsize=ticks_sz)
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10
            ax.zaxis.labelpad = 12

            plotSphere(
                ax,
                np.array([0, 0, 0]),
                env.omega_max,
                "red",
                0.2,
                # "$\| \omega \|$ Constraint",
                "$\mathcal{C}_{\\rm S} $",
            )

            backup_coeffs = (1 / (2 * env.gamma)) * np.diag(env.J)
            plotEllipsoid(ax, backup_coeffs, "cyan", 0.2, "$\mathcal{C}_{\\rm B} $")
            rtapoints = int(math.floor(env.backupTime / env.del_t))
            max_radius = 0.8 * env.omega_max
            for axis in "xyz":
                getattr(ax, "set_{}lim".format(axis))((-max_radius, max_radius))

            ax.plot(x1_v, x2_v, x3_v, color="red", linewidth=lwp, label=None)
            ax.plot(x1, x2, x3, color="blue", linewidth=lwp, label=None)

            if intervening:
                lists = splitfun(intervening, 2)
                for i in range(len(lists)):
                    if i == 0:
                        # label = "Safety Intervention"
                        label = None
                    else:
                        label = None
                    ax.plot(
                        x1[lists[i]],
                        x2[lists[i]],
                        x3[lists[i]],
                        color="magenta",
                        linewidth=lwp * 1.03,
                        label=label,
                    )

            n_etSphere = 3
            if env.backupTrajs:
                for i in np.arange(0, len(env.backupTrajs), step=2):
                    if i == 0:
                        label = r"$\boldsymbol{\phi}^{n}_{\rm b} (\tau, \boldsymbol{\omega})$"
                    else:
                        label = None

                    x_c = env.backupTrajs[i][:, 0]
                    y_c = env.backupTrajs[i][:, 1]
                    z_c = env.backupTrajs[i][:, 2]

                    if (
                        env.robust
                        and plot_backupSpheres
                        and i % n_etSphere == 0
                        and i != 0
                    ):
                        for j in np.arange(0, rtapoints, step=4):
                            plotSphere_epsilon(
                                ax,
                                np.array([x_c[j], y_c[j], z_c[j]]),
                                env.delta_array[j],
                                "gray",
                                0.1,
                                None,
                            )

                    plt.plot(
                        x_c,
                        y_c,
                        z_c,
                        color="cyan",
                        linewidth=lwp * 0.75,
                        label=label,
                    )
            ax.plot(
                x1[0],
                x2[0],
                x3[0],
                "k*",
                markersize=8,
                label=r"$\boldsymbol{\omega_0}$",
            )
            # ax.view_init(elev=29, azim=128)
            ax.view_init(elev=11, azim=136)

            norms_vanilla = np.zeros(x_vanilla.shape[1])
            for i in range(x_vanilla.shape[1]):
                norms_vanilla[i] = np.linalg.norm(x_vanilla[:, i], 2)

            viol_idxs = norms_vanilla > env.omega_max
            viol_locations = x_vanilla[:, viol_idxs]

            # Label violations
            text_pos = [0, -1.04, 0.95]
            ax.text(
                text_pos[0],
                text_pos[1],
                text_pos[2],
                "safety \n violations",
                # "violation",
                fontsize=legend_sz,
                # bbox=dict(facecolor="none", edgecolor="black", pad=10.0),
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            buff = 0.02
            buff2 = 0.05
            arrow1_dir = viol_locations[:, 50] + [-buff2, -buff2, 0]
            arrow_o = [text_pos[0] - buff, text_pos[1] + buff, text_pos[2] - buff]
            arrow_lw = 1.4

            # Arrow 1
            a = Arrow3D(
                [arrow_o[0], arrow1_dir[0]],
                [arrow_o[1], arrow1_dir[1]],
                [arrow_o[2], arrow1_dir[2]],
                mutation_scale=20,
                lw=arrow_lw,
                arrowstyle="-|>",
                color="k",
            )
            ax.add_artist(a)

            arrow2_dir = viol_locations[:, 100] + [0, 0, buff2]
            arrow_o = [
                text_pos[0] - 1 * buff,
                text_pos[1] - buff,
                text_pos[2] - 1 * buff,
            ]
            # Arrow 2
            a = Arrow3D(
                [arrow_o[0], arrow2_dir[0]],
                [arrow_o[1], arrow2_dir[1]],
                [arrow_o[2], arrow2_dir[2]],
                mutation_scale=20,
                lw=arrow_lw,
                arrowstyle="-|>",
                color="k",
            )
            ax.add_artist(a)

            if p1:
                ax.legend(
                    fontsize=legend_sz,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.06),
                    fancybox=True,
                    shadow=True,
                    ncol=2,
                )
                plt.tight_layout(pad=3)
                if save_plots:
                    plt.savefig("plots/3d_plot.png", dpi=400)
            else:
                ax.legend(
                    fontsize=legend_sz,
                    loc="center left",
                    bbox_to_anchor=(-0.163, 0.5),
                    fancybox=True,
                    shadow=True,
                )
                plt.tight_layout()
                if save_plots:
                    dpi = 225
                    # dpi = 425
                    path = "plots/3d_plot_2_comparison.png"
                    plt.savefig(path, dpi=dpi)
                    im = Image.open(path)
                    width, height = im.size
                    im1 = im.crop((0, (1 - dhr) * height, dwr * width, dhr * height))
                    im1.save("plots/3d_plot_crop_comparison.png")

        if norm_u_plot:

            xaxis_sz = 19
            legend_sz = 17

            fig = plt.figure(figsize=(11.69, 7), dpi=100)
            # fig = plt.figure(figsize=((dw/dh)*7, 7), dpi=100)
            ax = fig.add_subplot(2, 1, 1)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)

            plt.axhline(env.omega_max, color="k", linestyle="--")
            y1_fill = np.ones(x.shape[1]) * 0
            y2_fill = np.ones(x.shape[1]) * env.omega_max
            ax.fill_between(
                t_span,
                y1_fill,
                y2_fill,
                color=(240 / 255, 255 / 255, 240 / 255),  # Green, safe set
            )
            ax.fill_between(
                t_span,
                y2_fill,
                2 * env.omega_max * np.ones(x.shape[1]),
                color=(255 / 255, 230 / 255, 230 / 255),  # Red, unsafe set
            )

            norms_vanilla = np.zeros(x_vanilla.shape[1])
            for i in range(x_vanilla.shape[1]):
                norms_vanilla[i] = np.linalg.norm(x_vanilla[:, i], 2)

            plt.plot(
                t_span,
                norms_vanilla,
                color="red",
                linewidth=lwp,
                # label=None,
                label="bCBF-QP",
            )

            if env.backupTrajs:
                norms = []
                for j in range(len(env.backupTrajs)):
                    norm = np.zeros(len(env.backupTrajs[j]))
                    for z in range(len(env.backupTrajs[j])):
                        norm[z] = np.linalg.norm(env.backupTrajs[j][z], 2)
                    norms.append(norm)

                # Backup trajectory points
                tmax_b = env.backupTime
                rtapoints = int(math.floor(tmax_b / env.del_t))
                tspan_backup = np.linspace(0, env.backupTime - env.del_t, rtapoints)
                for i, norm in enumerate(norms):
                    if i == 0:
                        # label = "Nominal Backup Flow"
                        # label = r"$\boldsymbol{\phi}^{n}_{\rm b} (\tau, \boldsymbol{\omega})$"
                        label = None
                    else:
                        label = None
                    plt.plot(
                        tspan_backup + env.del_t * i * env.backup_save_N,
                        norm,
                        color="cyan",
                        linewidth=lwp * 0.8,
                        label=label,
                    )

            norms = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                norms[i] = np.linalg.norm(x[:, i], 2)

            plt.plot(
                t_span,
                norms,
                color="blue",
                linewidth=lwp,
                # label=None,
                label="DR-bCBF-QP",
                # label="Ours",
            )

            ax.set_xlim([0, t_span[-1]])
            ax.set_ylim([0, env.omega_max * 1.05])
            if intervening:
                lists = splitfun(intervening, 2)
                for i in range(len(lists)):
                    if i == 0:
                        # label = "Safety Intervention"
                        label = None
                    else:
                        label = None
                    ax.plot(
                        t_span[lists[i]],
                        norms[lists[i]],
                        color="magenta",
                        linewidth=lwp * 1.03,
                        label=label,
                    )

            unsafe_i = norms > env.omega_max
            if any(unsafe_i):
                plt.scatter(
                    t_span[unsafe_i],
                    norms[unsafe_i],
                    color="red",
                    marker="x",
                    s=40,
                    label="Violation",
                    zorder=10,
                )

            plt.ylabel(
                r"$\| \boldsymbol{\omega} \|$" + "($\\frac{\\rm rad}{\\rm s}$)",
                fontsize=xaxis_sz,
            )
            ax.legend(
                fontsize=legend_sz,
                # fancybox=True,
                # shadow=True,
                loc="lower right",
            )
            plt.grid(True)

            ax = fig.add_subplot(2, 1, 2)
            for i in range(np.shape(u_p)[0]):
                ax.grid(True)
                plt.xticks(fontsize=ticks_sz)
                plt.yticks(fontsize=ticks_sz)
                ax.set_xlim([0, t_span[-1]])  # Adjust the x-axis limits
                ax.set_ylim([1.05 * bounds[i][0], 1.05 * bounds[i][1]])

                if i == 0:
                    color = "#231942"
                elif i == 1:
                    color = "#9F86C0"
                else:
                    color = "#E0B1CB"

                ax.plot(
                    t_span_u,
                    u_p[i, 1:],
                    "--",
                    color=color,
                    # label="$u_{\\rm des," + f"{i+1}" + "}$",
                    label="$u_{\\rm p," + f"{i+1}" + "}$",
                    linewidth=lwp,
                )
                ax.plot(
                    t_span_u,
                    u_act[i, 1:],
                    "-",
                    color=color,
                    # label="$u_{\\rm act," + f"{i+1}" + "}$",
                    label="$u_{\\rm safe," + f"{i+1}" + "}$",
                    linewidth=lwp,
                )
                if i == 0:
                    if title_flag:
                        ax.set_title("Control Inputs vs. Time", fontsize=title_sz)
                    ax.set_ylabel("control torque (Nm)", fontsize=xaxis_sz)

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=3,
                # fancybox=True,
                fontsize=legend_sz,
                # shadow=True,
            )

            plt.xlabel("time (s)", fontsize=xaxis_sz)

            if save_plots:
                dpi = 500  # 250
                plt.savefig(
                    "plots/norm_u_plot_comparison.pdf", dpi=dpi, bbox_inches="tight"
                )

        if show_plots:
            plt.show()
