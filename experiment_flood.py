import csv
import numpy as np
import pandas as pd

import config
from main import run_simulation, create_gif_from_frames
from environment import LinearShiftEnvironment
from flood import ShockEnvironment
from population import Population
from mutation import IsotropicMutation
from selection import TwoStageSelection
from reproduction import AsexualReproduction
from visualization import plot_population, plot_frame, plot_stats

def make_population():
    return Population(
        size=config.N,
        n_dim=config.n,
        init_scale=config.init_scale,
        alpha_init=config.alpha0,
    )


def make_selection():
    return TwoStageSelection(
        sigma=config.sigma,
        threshold=config.threshold,
        N=config.N,
    )


def make_mutation():
    return IsotropicMutation(
        mu=config.mu,
        mu_c=config.mu_c,
        xi=config.xi,
    )


def make_reproduction():
    return AsexualReproduction()



'''
for both scenarios (naive and pre-adapted), stats are returned
'''

#=================================================================
#naive - no adaptation under linear shift environment
#=================================================================
def run_naive(seed, frames_dir=None, verbose=False):
    np.random.seed(seed)

    pop = make_population()

    env = ShockEnvironment(
        alpha_init=config.alpha0.copy(),
        c=config.c.copy(),
        delta=config.delta,
        T_shock=config.T_shock,
        sigma_shock=config.sigma_shock,
    )

    stats = run_simulation(
        population=pop,
        environment=env,
        selection_strategy=make_selection(),
        reproduction_strategy=make_reproduction(),
        mutation_strategy=make_mutation(),
        max_generations=config.flood_generations,
        frames_dir=frames_dir,
        verbose=verbose,
    )
    return stats



#=================================================================
#pre-adapted
#=================================================================


def run_pre_adapted(seed, frames_dir=None, verbose=False):
    np.random.seed(seed)

    pop = make_population()

    #preadaptation under linear shift environment (no floods)
    env_pre = LinearShiftEnvironment(
        alpha_init=config.alpha0.copy(),
        c=config.c.copy(),
        delta=config.delta,
    )

    run_simulation(
        population=pop,
        environment=env_pre,
        selection_strategy=make_selection(),
        reproduction_strategy=make_reproduction(),
        mutation_strategy=make_mutation(),
        max_generations=config.pre_adaptation_generations,
        frames_dir=None,
        verbose=False,
    )

    #new optimum after pre-adatation
    alpha_after_pre = env_pre.get_optimal_phenotype().copy()

    #change of environment - introduction of floods
    env_flood = ShockEnvironment(
        alpha_init=alpha_after_pre,
        c=config.c.copy(),
        delta=config.delta,
        T_shock=config.T_shock,
        sigma_shock=config.sigma_shock,
    )

    stats = run_simulation(
        population=pop,
        environment=env_flood,
        selection_strategy=make_selection(),
        reproduction_strategy=make_reproduction(),
        mutation_strategy=make_mutation(),
        max_generations=config.flood_generations,
        frames_dir=frames_dir,
        verbose=verbose,
    )
    return stats


def main():
    if config.seed is not None:
        np.random.seed(config.seed)

    # Naive simulation
    print("Starting NAIVE simulation...\n")

    frames_dir = "frames_naive"

    stats_naive = run_naive(
        seed=5,
        frames_dir=frames_dir,
        verbose=True,
    )

    print(f"\nNAIVE:\n{stats_naive.summary()}")
    print("\n Creating GIF - NAIVE...")
    create_gif_from_frames(frames_dir, "naive_simulation.gif")
    print("GIF saved as naive_simulation.gif")
    plot_stats(stats_naive, save_path="naive_stats.png", show_plot=False)
    print("Plot saved as naive_stats.png")


    # Pre adapted simulation
    print("\nStarting PRE_ADAPTED simulation...\n")

    frames_dir = "frames_pre_adapted"

    stats_pre = run_pre_adapted(
        seed=5,
        frames_dir=frames_dir,
        verbose=True,
    )

    print(f"\n PRE_ADAPTED:\n{stats_pre.summary()}")
    print("\n Creating GIF - PRE_ADAPTED...")
    create_gif_from_frames(frames_dir, "pre_adapted_simulation.gif")
    print("GIF saved as pre_adapted_simulation.gif")
    plot_stats(stats_pre, save_path="pre_adapted_stats.png", show_plot=False)
    print("Plot saved as pre_adapted_stats.png")
    
if __name__ == "__main__":
    main()


