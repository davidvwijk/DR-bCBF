"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Module containing control laws.

"""

import numpy as np


class Control:
    def setupControl(
        self,
    ):
        # Control limits
        self.u_max = 1  # [Nm]
        self.u_bounds = ((-self.u_max, self.u_max),) * 3

        sig_min = min(np.linalg.eigvals(self.J))
        sig_max = max(np.linalg.eigvals(self.J))

        self.k_b = 1 * (
            np.sqrt((2 * self.gamma) / sig_min)
            * (sig_max * self.dw_max)
            * (1 / (2 * self.gamma))
        )

        self.k_b_tanh = 30

        # Robustness constants
        self.L_cl = self.k_b
        self.sup_fcl = self.k_b * self.omega_max

    def primaryControl(self, x, t):
        """
        Primary controller producing desired control at each step.

        """
        u_des = self.u_max * np.sin(
            np.array([t / 2, t / 2 - np.pi / 4, t / 4 + np.pi / 4])
        )
        return u_des

    def backupControl_tanh(self, x):
        """
        Safe backup controller.

        """
        u_b = self.u_max * np.tanh(-self.k_b_tanh * self.J @ x)
        return u_b

    def backupControl_robustLyap(self, x):
        """
        Safe backup controller which has been proven to be robust to disturbances using a Lyapunov analysis.

        """
        return self.fastCross(x, self.J @ x) - self.k_b * self.J @ x
