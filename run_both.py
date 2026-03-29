from flood import  ShockEnvironment
import main

import os
import numpy as np

import config
from environment import LinearShiftEnvironment
from population import Population
from mutation import IsotropicMutation
from selection import TwoStageSelection
from reproduction import AsexualReproduction
from visualization import plot_population, plot_frame, plot_stats
from stats import SimulationStats

'''
this file runs main.py - baseline and flood - experimental
'''

#============================================================================
#running baseline
#==========================================================================


if __name__ == "__main__":
    print("Uru")
    main.main()


#running the flood
    # env = ShockEnvironment(
    #     alpha_init=config.alpha0,
    #     c=config.c,
    #     delta=config.delta,
    #     T_shock=config.T_shock,
    #     sigma_shock=config.sigma_shock
    # )










#with flood


# def main():
#     # --- Ziarno losowości (config.seed = None → inna symulacja za każdym razem) ---
#     if config.seed is not None:
#         np.random.seed(config.seed)

#     # --- Inicjalizacja komponentów - baseline
   
#     env = LinearShiftEnvironment(
#         alpha_init=config.alpha0,
#         c=config.c,
#         delta=config.delta
#     )

#     pop = Population(
#         size=config.N,
#         n_dim=config.n,
#         init_scale=config.init_scale,
#         alpha_init=config.alpha0,   # populacja startuje blisko alpha0, nie wokół zera
#     )
#     selection = TwoStageSelection(
#         sigma=config.sigma,
#         threshold=config.threshold,
#         N=config.N
#     )
#     reproduction = AsexualReproduction()
#     mutation = IsotropicMutation(
#         mu=config.mu,
#         mu_c=config.mu_c,
#         xi=config.xi,
#     )

#     # --- Uruchomienie symulacji ---
#     print("Rozpoczynam symulację...\n")
#     frames_dir = "frames"
#     stats = run_simulation(
#         population=pop,
#         environment=env,
#         selection_strategy=selection,
#         reproduction_strategy=reproduction,
#         mutation_strategy=mutation,
#         frames_dir=frames_dir,
#         verbose=True,
#     )

#     print(f"\n{stats.summary()}")

#     # --- GIF ---
#     print("\nTworzenie GIF-a...")
#     create_gif_from_frames(frames_dir, "simulation.gif")
#     print("GIF zapisany jako simulation.gif")

#     # --- Wykres statystyk ---
#     plot_stats(stats, save_path="simulation_stats.png", show_plot=False)
#     print("Wykres statystyk zapisany jako simulation_stats.png")


# if __name__ == "__main__":
#     main()
