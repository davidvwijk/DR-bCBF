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

import numpy as np
import time
import math
import quadprog

# from qpsolvers import solve_qp


class Constraint:
    def alpha(self, x):
        """
        Strengthening function.
        Must be strictly increasing with the property that alpha(x=0) = 0.
        For alpha(·) = η*h(x) for η>0, higher values of η result in less conservative barrier constraints.

        """
        return 2 * x

    def h_x(self, x):
        """
        Constraint function, or control barrier function.

        """
        h = self.omega_max**2 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2
        return h

    def grad_h(self, x):
        """
        Gradient of constraint function.

        """
        return -2 * x

    def hb_x(self, x):
        """
        Reachability constraint.

        """

        return self.gamma - 0.5 * x.T @ self.J @ x

    def grad_hb(self, x):
        """
        Gradient of reachability constraint.

        """
        return -x @ self.J

    def alpha_b(self, x):
        """
        Strengthening function.
        Must be strictly increasing with the property that alpha(x=0) = 0.
        For alpha(·) = η*h(x) for η>0, higher values of η result in less conservative barrier constraints.

        """
        return 10 * x


class ASIF(Constraint):
    """
    Class containing necessary functions for active set invariance filtering.

    """

    def setupASIF(
        self,
    ):
        # Backup properties
        self.backupTime = 1.75  # [sec]
        self.backupTrajs = []
        self.backup_save_N = 5  # saves every N backup trajectory (for plotting)

        # Constraint
        self.omega_max = 1  # [rad/sec]
        self.violation_vals = []

        # Tightening constants
        self.Lh_const = 2 * self.omega_max
        self.Lhb_const = self.omega_max * max(np.linalg.eigvals(self.J))
        self.gamma = 2
        self.delta_array = []

    def asif(self, x, u_des):
        """
        Implicit active set invariance filter (ASIF) using QP.

        """

        # QP objective function
        M = np.eye(3)
        q = u_des

        # QP actuation constraints
        G = np.vstack((np.eye(3), -np.eye(3)))
        h = -self.u_max * np.ones(6)

        # Backup trajectory points
        rtapoints = int(math.floor(self.backupTime / self.del_t))

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x

        # Sensitivity matrix tracking array
        S = np.zeros((lenx, lenx, rtapoints))
        S[:, :, 0] = np.eye(lenx)

        # Simulate system under backup control law
        new_x = np.concatenate((x, S[:, :, 0].flatten()))

        tic = time.perf_counter()

        backupFlow = self.integrateStateBackup(
            new_x,
            np.arange(0, self.backupTime, self.del_t),
            self.int_options,
        )

        toc = time.perf_counter()
        backupComputeTime = toc - tic

        phi[:, :] = backupFlow[:lenx, :].T
        S[:, :, :] = backupFlow[lenx:, :].reshape(lenx, lenx, rtapoints)

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        fx_0 = self.f_x(x)
        gx_0 = self.g_x(x)

        # Construct barrier constraint for each point along trajectory
        for i in range(rtapoints):
            # Compute h(phi)
            h_phi = self.h_x(phi[i, :])
            gradh_phi = self.grad_h(phi[i, :])
            g_temp_i = gradh_phi.T @ S[:, :, i] @ gx_0

            epsilon = 0
            robust_grad = 0
            if self.robust:
                t = self.del_t * i

                # Contraction Bound
                delta_t = (self.dw_max / self.k_b) * (1 - np.exp(-self.k_b * t))

                # Norm bound tightening term
                epsilon = self.Lh_const * delta_t

                # Discretization tightening constant
                mu_d = (self.del_t / 2) * self.Lh_const * (self.sup_fcl + self.dw_max)

                # Robustification term
                robust_grad = np.linalg.norm(gradh_phi @ S[:, :, i]) * self.dw_max

                # Store only the first time
                if len(self.delta_array) < rtapoints:
                    self.delta_array.append(delta_t)
            else:
                # Discretization tightening constant required for vanilla bCBF
                mu_d = (self.del_t / 2) * self.Lh_const * self.sup_fcl

            h_temp_i = (
                -(gradh_phi @ S[:, :, i] @ fx_0 + self.alpha(h_phi - epsilon - mu_d))
                + robust_grad
            )

            if i == 0:
                g_temp = g_temp_i
                h_temp = h_temp_i
            else:
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

            # Make sure last point is in the backup set
            if i == rtapoints - 1:

                hb_phi = self.hb_x(phi[i, :])
                gradhb_phi = self.grad_hb(phi[i, :])

                robust_grad_b = 0
                grad_epsilon_b = np.zeros(3)
                epsilon_b = 0
                if self.robust:
                    # Contraction Bound
                    delta_T = delta_t
                    epsilon_b = np.linalg.norm(phi[i, :] @ self.J) * delta_T

                    if np.linalg.norm(phi[i, :]) != 0:
                        grad_epsilon_b = (
                            (self.J @ np.transpose(self.J) @ phi[i, :])
                            / (
                                np.sqrt(
                                    phi[i, :].T
                                    @ self.J
                                    @ np.transpose(self.J)
                                    @ phi[i, :]
                                )
                            )
                        ) * delta_T
                    else:
                        grad_epsilon_b = np.zeros(3)

                    robust_grad_b = (
                        np.linalg.norm(gradhb_phi @ S[:, :, i] - grad_epsilon_b)
                        * self.dw_max
                    )

                h_temp_i = (
                    -(
                        (gradhb_phi @ S[:, :, i] - grad_epsilon_b) @ fx_0
                        + self.alpha_b(hb_phi - epsilon_b)
                    )
                    + robust_grad_b
                )
                g_temp_i = (gradhb_phi.T @ S[:, :, i] - grad_epsilon_b) @ gx_0

                # Append constraint
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

        # Append constraints
        G = np.vstack([G, g_temp])
        h = np.vstack([h.reshape((-1, 1)), h_temp])

        # Solve QP
        d = h.reshape((len(h),))
        try:
            tic = time.perf_counter()
            soltn = quadprog.solve_qp(M, q, G.T, d, 0)
            active_constraint = soltn[5]
            u_act = soltn[0]
            toc = time.perf_counter()
            solverComputeTime = toc - tic
        except:
            # u_act = self.backupControl(x)
            u_act = u_des
            solverComputeTime = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, safety filter is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solverComputeTime, backupComputeTime
