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
    ) -> None:

        # Control limits
        self.u_max = 1
        self.u_bounds = ((-self.u_max, self.u_max),)

    def primaryControl(self, x_curr, t):
        """
        Primary controller producing desired control at each step.

        """
        u_des = self.u_max
        return u_des

    def backupControl(self, x):
        """
        Safe backup controller. Constant for this application.

        """
        u_b = -self.u_max
        return u_b
