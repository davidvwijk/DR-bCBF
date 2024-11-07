"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Disturbance-Robust Backup Control Barrier Functions (DR-bCBF) code base.

© 2024 David van Wijk <davidvanwijk@tamu.edu>

---------------------------------------------------------------------------

Calls Simulation class for the standard bCBF approach and our DR-bCBF approach, running each and comparing the two.

"""

from main_sim import Simulation
from plotting import Plotter

if __name__ == "__main__":

    show_plots = True
    save_plots = False

    # Run baseline (vanilla backup CBF)
    print("Running simulation with bCBF-QP")
    env1 = Simulation(
        safety_flag=True,
        verbose=False,
        robust=False,
        dw_bool=True,
    )
    (
        x_full_vanilla,
        total_steps,
        u_des_full_vanilla,
        u_act_full_vanilla,
        _,
        _,
        _,
    ) = env1.sim()

    # Run ours (disturbance-robust backup CBF)
    print("Running simulation with DR-bCBF-QP")
    env2 = Simulation(
        safety_flag=True,
        verbose=False,
        robust=True,
        dw_bool=True,
    )
    (
        x_full,
        total_steps,
        u_des_full,
        u_act_full,
        intervened,
        _,
        _,
    ) = env2.sim()

    Plotter.comparison_plotter(
        x_full_vanilla,
        x_full,
        u_act_full,
        intervened,
        u_des_full,
        env2,
        sphere_plot=True,
        latex_plots=True,
        save_plots=save_plots,
        show_plots=show_plots,
        norm_u_plot=True,
    )
