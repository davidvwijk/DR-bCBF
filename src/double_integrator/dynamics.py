"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains functions required for propagating dynamics of double integrator.

"""

import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp


class Dynamics:
    def setupDynamics(
        self,
    ) -> None:

        # A and B matrices constant
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([0, 1])

        # Integration options
        self.int_options = {"rtol": 1e-9, "atol": 1e-9}

        # Simulation data
        self.del_t = 0.02  # [sec]
        self.total_steps = int(5 / self.del_t) + 2
        self.curr_step = 0

        # Initial conditions
        self.x0 = np.array([-1.5, 0])

        # Disturbances
        self.dw_max = 0.08

        # Constant
        max_vel = 2  # based on starting x
        self.sup_fcl = np.sqrt(max_vel**2 + 1)

    def propMain(self, t, x, u, dist, args):
        """
        Propagation function for dynamics with disturbance and STM if applicable.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) * u + dist
        if len(x) > lenx:
            # Construct F
            F = self.computeJacobianSTM(x[:lenx])

            # Extract STM & reshape
            STM = x[lenx:].reshape(lenx, lenx)
            dSTM = F @ STM

            # Reshape back to column
            dSTM = dSTM.reshape(lenx**2)
            dx[lenx:] = dSTM

        return dx

    def computeJacobianSTM(self, x):
        """
        Compute Jacobian of dynamics.
        """

        jac = self.A
        return jac

    def f_x(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        f = self.A @ x
        return f

    def g_x(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        g = self.B
        return g

    def integrateState(self, x, u, t_step, dist, options):
        """
        State integrator using propagation function.

        """
        t_step = (0.0, t_step)
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMain(t, x, u, dist, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
        )
        x = soltn.y[:, -1]
        return x

    def propMainBackup(self, t, x, args):
        """
        Propagation function for backup dynamics and STM if applicable.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.A @ x[:lenx] + self.B * self.backupControl(x[:lenx])

        # Construct F
        F = self.A

        # Extract STM & reshape
        STM = x[lenx:].reshape(lenx, lenx)
        dSTM = F @ STM

        # Reshape back to column
        dSTM = dSTM.reshape(lenx**2)
        dx[lenx:] = dSTM

        return dx

    def integrateStateBackup(self, x, tspan_b, options):
        """
        State integrator using propagation function.

        """
        t_step = (0.0, tspan_b[-1])
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMainBackup(t, x, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
            t_eval=tspan_b,
        )
        x = soltn.y[:, :]
        return x

    def disturbanceFun(self, t, x, u, args):
        """
        Process disturbance function, norm bounded by dw_max.

        """
        dist_t = np.array([1, 1])
        return (
            (dist_t / (np.linalg.norm(dist_t))) * self.dw_max
            if np.linalg.norm(dist_t) != 0
            else dist_t
        )
