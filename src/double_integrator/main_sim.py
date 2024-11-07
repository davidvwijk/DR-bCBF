"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module runs full simulation for double integrator example.

"""

import numpy as np
from safety import ASIF
from control import Control
from dynamics import Dynamics
from plotting import Plotter


class Simulation(ASIF, Control, Dynamics):
    def __init__(
        self,
        safety_flag=True,
        verbose=True,
        robust=False,
        dw_bool=False,
    ) -> None:

        self.setupDynamics()
        self.setupASIF()
        self.setupControl()

        self.verbose = verbose
        self.safety_flag = safety_flag
        self.robust = robust
        self.dw_bool = dw_bool

    def checkViolation(self, x_curr):
        """
        Check for safety violation.

        """
        h_funs = [lambda x: self.h1_x(x)]
        for i in range(len(h_funs)):
            h_x = h_funs[i]
            if h_x(x_curr) < 0:
                if self.verbose:
                    print(f"Safety violation, constraint {i+1}")

    def sim(self):
        """
        Simulates trajectory for pre-specified number of timesteps and performs
        point-wise safety-critical control modifications if applicable.

        """
        x0 = self.x0
        total_steps = self.total_steps

        # Tracking variables
        x_full = np.zeros((len(x0), total_steps))
        u_act_full = np.zeros((len(self.u_bounds), total_steps))
        u_des_full = np.zeros((len(self.u_bounds), total_steps))
        solver_times, avg_solver_t, max_solver_t, intervened = [], [], [], []
        x_full[:, 0] = x0
        x_curr = x0

        # Disturbance variables
        self.dw_full = np.zeros((len(x0), total_steps))

        # Main loop
        for i in range(1, total_steps):
            t = self.curr_step * self.del_t

            # Generate desired control
            u_des = self.primaryControl(x_curr, t)
            u_des_full[:, i] = u_des

            # If safety check on, monitor control
            if self.safety_flag:
                u, boolean, sdt = self.asif(x_curr, u_des)
                solver_times.append(sdt)
                if boolean:
                    intervened.append(i)
            else:
                u = u_des

            # Compute disturbance vectors (if applicable)
            if self.dw_bool:
                dw = self.disturbanceFun(t, x_curr, u, {})
                self.dw_full[:, i] = dw
            else:
                dw = np.zeros(len(x0))

            u_act_full[:, i] = u

            # Propagate states with control and disturbances (if applicable)
            x_curr = self.integrateState(
                x_curr,
                u,
                self.del_t,
                dw,
                self.int_options,
            )
            x_full[:, i] = x_curr

            self.curr_step += 1
            if self.h1_x(x_curr) < 0:
                print("Crashed")

        if self.verbose and self.safety_flag:
            solver_times = [
                i for i in solver_times if i is not None
            ]  # Remove Nones which can be due to QP failures
            if solver_times:
                avg_solver_t = 1000 * np.average(solver_times)  # in ms
                max_solver_t = 1000 * np.max(solver_times)  # in ms
                print(f"Average solver time: {avg_solver_t:0.4f} ms")
                print(f"Maximum single solver time: {max_solver_t:0.4f} ms")

        return (
            x_full,
            total_steps,
            u_des_full,
            u_act_full,
            intervened,
            avg_solver_t,
            max_solver_t,
        )


if __name__ == "__main__":

    env = Simulation(
        safety_flag=True,
        verbose=True,
        robust=True,
        dw_bool=True,
    )
    print(
        "Running simulation with parameters:",
        "Safety:",
        env.safety_flag,
        "| Robustness:",
        env.robust,
        "| Process dist:",
        env.dw_bool,
    )

    (
        x_full,
        total_steps,
        u_des_full,
        u_act_full,
        intervened,
        avg_solver_t,
        max_solver_t,
    ) = env.sim()

    Plotter.plotter(
        x_full,
        u_act_full,
        intervened,
        u_des_full,
        env,
        phase_plot_1=False,
        phase_plot_2a=True,
        phase_plot_2b=False,
        control_plot=True,
        latex_plots=True,
        save_plots=False,
        show_plots=True,
        legend_flag=True,
    )
