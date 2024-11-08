"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module contains functions required for propagating dynamics of rigid body spacecraft.

"""

import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp


class Dynamics:
    def setupDynamics(
        self,
    ) -> None:
        # Intertial properties of spacecraft
        self.J = np.diag([12, 12, 5])  # [kgm^2]
        self.invJ = np.linalg.inv(self.J)

        # Integration options
        self.int_options = {"rtol": 1e-6, "atol": 1e-6}

        # Simulation data
        self.del_t = 0.05  # [sec]
        self.total_steps = 500
        self.curr_step = 0

        # Initial conditions
        self.x0 = np.array([0.0, 0.0, 0.0])  # [rad/sec]

        # Disturbance
        self.dw_max = 0.1  # [rad/sec^2]

    def propMain(self, t, x, u, dist, args):
        """
        Propagation function for dynamics with disturbance and STM if applicable.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.invJ @ u + dist
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

    def skew(self, x):
        """
        Constructs skew symmetric matrix.

        """
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    def fastCross(self, x1, x2):
        """
        Calculates cross product, cross(x1,x2). Faster than using np.cross() for this application.

        Args:
            x1 (numpy.ndarray): nx1 vector.
            x2 (numpy.ndarray): nx1 vector.

        Returns:
            numpy.ndarray: nxn matrix.
        """
        return self.skew(x1) @ x2

    def computeJacobianSTM(self, x):
        """
        Compute Jacobian of dynamics.

        """
        J = self.J
        jac = -self.invJ @ (self.skew(x) @ J - self.skew(J @ x))
        return jac

    def f_x(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        f = self.invJ @ (self.fastCross(-x, self.J @ x))
        return f

    def f_x_jax(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.
        Jax implementation.

        """
        f = jnp.linalg.inv(self.J) @ (jnp.cross(-x, self.J @ x))
        return f

    def g_x(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        g = self.invJ
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
        Propagation function for backup dynamics and STM.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.invJ @ (
            (self.fastCross(-x[:lenx], self.J @ x[:lenx]))
            + self.backupControl_robustLyap(x[:lenx])
        )

        # Construct F
        F = self.computeJacobianSTM(x[:lenx])

        # Extract STM & reshape
        STM = x[lenx:].reshape(lenx, lenx)
        dSTM = F @ STM

        # Reshape back to column
        dSTM = dSTM.reshape(lenx**2)
        dx[lenx:] = dSTM

        return dx

    def integrateStateBackup(self, x, tspan_b, options):
        """
        Propagate backup flow over the backup horizon. Evaluate at discrete points.

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
        dist_t = np.sin(np.array([t / 2 + np.pi / 2, t / 2, t / 2 - np.pi / 2]))
        return (
            (dist_t / (np.linalg.norm(dist_t))) * self.dw_max
            if np.linalg.norm(dist_t) != 0
            else dist_t
        )
