"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains functions required for safety-critical control using control barrier functions.

"""

from matplotlib import testing
import numpy as np
import time
import math
import quadprog
from scipy.integrate import quad


class Constraint:
    def alpha(self, x):
        """
        Strengthening function.

        """
        return 15 * x + x**3

    def alpha_b(self, x):
        """
        Strengthening function for reachability constraint.

        """
        return 10 * x

    def h1_x(self, x):
        """
        Safety constraint.

        """
        h = -x[0]
        return h

    def grad_h1(self, x):
        """
        Gradient of safety constraint.

        """
        g = np.array([-1, 0])
        return g

    def hb_x(self, x):
        """
        Reachability constraint.

        """
        hb = -x[1]
        return hb

    def grad_hb(self, x):
        """
        Gradient of reachability constraint.

        """
        gb = np.array([0, -1])
        return gb


class ASIF(Constraint):
    def setupASIF(
        self,
    ) -> None:

        # Backup properties
        self.backupTime = 1.5  # [sec] (total backup time)
        self.backupTrajs = []
        self.backup_save_N = 5  # saves every N backup trajectory (for plotting)
        self.delta_array = [0]

        # Tightening constants
        self.Lh_const = 1
        self.Lhb_const = 1
        self.L_cl = 1  # Lipschitz constant of closed-loop dynamics

    def asif(self, x, u_des):
        """
        Implicit active set invariance filter (ASIF) using QP.

        """

        # QP objective function
        M = np.eye(2)
        q = np.array(
            [u_des, 0.0]
        )  # Need to append the control with 0 to get at least 2 dimensions

        # Control constraints
        G = [[1.0, 0.0], [-1.0, 0.0]]
        h = [-self.u_max, -self.u_max]

        # Total backup trajectory time
        tmax_b = self.backupTime

        # Backup trajectory points
        rtapoints = int(math.floor(tmax_b / self.del_t)) + 1

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x

        # Sensitivity matrix tracking array
        S = np.zeros((lenx, lenx, rtapoints))
        S[:, :, 0] = np.eye(lenx)

        # Simulate flow under backup control law
        new_x = np.concatenate((x, S[:, :, 0].flatten()))

        # backupFlow = self.integrateStateBackup(
        #     new_x,
        #     np.arange(0, self.backupTime, self.del_t),
        #     self.int_options,
        # )

        # phi[:, :] = backupFlow[:lenx, :].T
        # S[:, :, :] = backupFlow[lenx:, :].reshape(lenx, lenx, rtapoints)

        for i in range(rtapoints - 1):
            u_b = self.backupControl(phi[i, :])  # Constant for this application

            # Compute the nominal flow (no disturbances)
            new_x = self.integrateState(
                new_x,
                u_b,
                self.del_t,
                np.zeros(2),
                self.int_options,
            )
            phi[i + 1, :] = new_x[0:lenx]
            S[:, :, i + 1] = new_x[lenx:].reshape((lenx, lenx))

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        fx_0 = self.f_x(x)
        gx_0 = self.g_x(x)

        # Construct barrier constraint for each point along trajectory
        for i in range(
            1, rtapoints
        ):  # Skip first point because of relative degree issue (general problem with BaCBFs)

            h_phi = self.h1_x(phi[i, :])
            gradh_phi = self.grad_h1(phi[i, :])
            g_temp_i = gradh_phi.T @ S[:, :, i] @ gx_0

            epsilon = 0
            robust_grad = 0
            if self.robust:
                t = self.del_t * i

                # Gronwall bound
                delta_t = (self.dw_max / self.L_cl) * (np.exp(self.L_cl * t) - 1)

                # Linear systems analysis bound (tighter than GW)
                def integrand(x, t):
                    return np.sqrt(
                        0.5 * ((t - x) ** 2 + 2 + (t - x) * np.sqrt((t - x) ** 2 + 4))
                    )

                # delta_t = self.dw_max * quad(integrand, 0, t, args=(t))[0]

                # Tightening epsilon
                epsilon = self.Lh_const * delta_t

                # Discretization tightening constant
                mu_d = (self.del_t / 2) * self.Lh_const * (self.sup_fcl + self.dw_max)

                # Robustness term
                robust_grad = np.linalg.norm(gradh_phi @ S[:, :, i]) * self.dw_max
            else:
                # Discretization tightening constant
                mu_d = (self.del_t / 2) * self.Lh_const * self.sup_fcl

            # Store only the first time
            if len(self.delta_array) < rtapoints:
                self.delta_array.append(delta_t)

            h_temp_i = (
                -(gradh_phi @ S[:, :, i] @ fx_0 + self.alpha(h_phi - epsilon - mu_d))
                + robust_grad
            )

            # Append constraint
            G.append([g_temp_i, 0])
            h.append(h_temp_i)

            # Make sure last point is in the backup set
            if i == rtapoints - 1:

                hb_phi = self.hb_x(phi[i, :])
                gradhb_phi = self.grad_hb(phi[i, :])

                robust_grad_b = 0
                epsilonT = 0
                if self.robust:
                    # Tightening epsilon
                    epsilonT = self.Lhb_const * delta_t

                    # Robustness term
                    robust_grad_b = (
                        np.linalg.norm(gradhb_phi @ S[:, :, i]) * self.dw_max
                    )

                h_temp_i = (
                    -(gradhb_phi @ S[:, :, i] @ fx_0 + self.alpha_b(hb_phi - epsilonT))
                    + robust_grad_b
                )
                g_temp_i = gradhb_phi.T @ S[:, :, i] @ gx_0

                # Append constraint
                G.append([g_temp_i, 0])
                h.append(h_temp_i)

        # Solve QP
        try:
            tic = time.perf_counter()
            sltn = quadprog.solve_qp(M, q, np.array(G).T, np.array(h), 0)
            u_act = sltn[0]
            active_constraint = sltn[5]
            toc = time.perf_counter()
            solver_dt = toc - tic
            u_act = u_act[0]  # Only extract scalar we need
        except:
            u_act = u_b
            solver_dt = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solver_dt
