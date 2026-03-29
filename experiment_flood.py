import csv
import numpy as np
import pandas as pd

import config
from main import run_simulation
from environment import LinearShiftEnvironment
from flood import ShockEnvironment
from population import Population
from mutation import IsotropicMutation
from selection import TwoStageSelection
from reproduction import AsexualReproduction


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
def run_naive(seed):
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
        frames_dir=None,
        verbose=False,
    )
    return stats



#=================================================================
#pre-adapted
#=================================================================


def run_pre_adapted(seed):
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
        frames_dir=None,
        verbose=False,
    )
    return stats


def main():
    results = []
    
    # zapis wynikow - do zmiany potem  
    with open("flood_results_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition",
            "seed",
            "last_generation",
            "final_mean_fitness",
            "final_distance_from_optimum",
            "final_phenotype_variance",
            "survived",
            "extinct_at",
        ])

        #running for 20 independent replicates
        for seed in range(20):
            stats_naive = run_naive(seed)
            stats_pre = run_pre_adapted(seed)

            for condition, stats in [
                ("naive", stats_naive),
                ("pre_adapted", stats_pre),
            ]:
                if stats.records:
                    last = stats.records[-1]

                    writer.writerow([
                        condition,
                        seed,
                        last.generation,
                        last.mean_fitness,
                        last.distance_from_optimum,
                        last.phenotype_variance,
                        stats.survived(),
                        stats.extinct_at,
                    ])

                    results.append({
                        "condition": condition,
                        "seed": seed,
                        "last_generation": last.generation,
                        "final_mean_fitness": last.mean_fitness,
                        "final_distance_from_optimum": last.distance_from_optimum,
                        "final_phenotype_variance": last.phenotype_variance,
                        "survived": stats.survived(),
                        "extinct_at": stats.extinct_at,
                    })

                    print(
                        f"{condition:12s} | seed={seed:2d} | "
                        f"Pokolenie {last.generation:4d} | "
                        f"fitness: {last.mean_fitness:.3f} | "
                        f"dist: {last.distance_from_optimum:.3f} | "
                        f"var: {last.phenotype_variance:.4f} | "
                        f"survived: {stats.survived()}"
                    )

                else:
                    writer.writerow([
                        condition,
                        seed,
                        None,
                        None,
                        None,
                        None,
                        False,
                        stats.extinct_at,
                    ])

                    results.append({
                        "condition": condition,
                        "seed": seed,
                        "last_generation": None,
                        "final_mean_fitness": None,
                        "final_distance_from_optimum": None,
                        "final_phenotype_variance": None,
                        "survived": False,
                        "extinct_at": stats.extinct_at,
                    })

    print("\nZapisano wyniki do flood_results_summary.csv")


    df = pd.DataFrame(results)

    summary = df.groupby("condition").agg(
        mean_fitness=("final_mean_fitness", "mean"),
        mean_distance=("final_distance_from_optimum", "mean"),
        mean_variance=("final_phenotype_variance", "mean"),
        survival_rate=("survived", "mean"),
        n_survived=("survived", "sum"),
        n_total=("survived", "count"),
    )

    print("\n=== SUMMARY ===")
    print(summary)

    summary.to_csv("summary_results.csv")
    print("\nZapisano summary_results.csv")


if __name__ == "__main__":
    main()
