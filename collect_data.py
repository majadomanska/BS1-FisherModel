import pickle
import csv
import numpy as np

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


def run_baseline(seed, delta_value):
    np.random.seed(seed)

    pop = make_population()
    env = LinearShiftEnvironment(
        alpha_init=config.alpha0.copy(),
        c=config.c.copy(),
        delta=delta_value,
    )

    stats = run_simulation(
        population=pop,
        environment=env,
        selection_strategy=make_selection(),
        reproduction_strategy=make_reproduction(),
        mutation_strategy=make_mutation(),
        max_generations=config.pre_adaptation_generations + config.flood_generations,
        frames_dir=None,
        verbose=False,
    )
    return stats


def run_naive_flood(seed, delta_value):
    np.random.seed(seed)

    pop = make_population()
    env = ShockEnvironment(
        alpha_init=config.alpha0.copy(),
        c=config.c.copy(),
        delta=delta_value,
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


def run_pre_adapted_flood(seed, delta_value):
    np.random.seed(seed)

    pop = make_population()

    # Faza 1: neutralne warunki (preadaptacja)
    env_pre = LinearShiftEnvironment(
        alpha_init=config.alpha0.copy(),
        c=config.c.copy(),
        delta=delta_value,
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

    # Faza 2: powodzie
    alpha_after_pre = env_pre.get_optimal_phenotype().copy()

    env_flood = ShockEnvironment(
        alpha_init=alpha_after_pre,
        c=config.c.copy(),
        delta=delta_value,
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
    # dwa warianty parametrów 
    variants = {
        "no_noise": 0.0,
        "with_noise": 0.02,
    }

    results = {}

    for variant_name, delta_value in variants.items():
        print(f"\n=== {variant_name} | delta = {delta_value} ===")

        results[variant_name] = {
            "baseline": [],
            "naive_flood": [],
            "pre_adapted_flood": [],
        }

        for seed in range(20):
            stats_baseline = run_baseline(seed, delta_value)
            stats_naive = run_naive_flood(seed, delta_value)
            stats_pre = run_pre_adapted_flood(seed, delta_value)

            results[variant_name]["baseline"].append(stats_baseline)
            results[variant_name]["naive_flood"].append(stats_naive)
            results[variant_name]["pre_adapted_flood"].append(stats_pre)

            print(f"done: {variant_name}, seed={seed}")

    # SAVE FULL OBJECTS (pickle)
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nZapisano pełne wyniki do results.pkl")

    # SAVE  CSV
    with open("results_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "variant",
            "condition",
            "seed",
            "last_generation",
            "final_fitness",
            "final_distance",
            "final_variance",
            "survived",
            "extinct_at",
        ])

        for variant_name, variant in results.items():
            for condition, stats_list in variant.items():
                for seed, stats in enumerate(stats_list):
                    if stats.records:
                        last = stats.records[-1]
                        writer.writerow([
                            variant_name,
                            condition,
                            seed,
                            last.generation,
                            last.mean_fitness,
                            last.distance_from_optimum,
                            last.phenotype_variance,
                            stats.survived(),
                            stats.extinct_at,
                        ])
                    else:
                        writer.writerow([
                            variant_name,
                            condition,
                            seed,
                            None,
                            None,
                            None,
                            None,
                            False,
                            stats.extinct_at,
                        ])

    print("Zapisano wyniki do results_summary.csv")

    print("\n=== PODSUMOWANIE ===")

    for variant_name, variant in results.items():
        print(f"\n--- {variant_name} ---")

        for condition, stats_list in variant.items():
            fitness = []
            distance = []
            variance = []
            survived = []

            for stats in stats_list:
                if stats.records:
                    last = stats.records[-1]
                    fitness.append(last.mean_fitness)
                    distance.append(last.distance_from_optimum)
                    variance.append(last.phenotype_variance)
                    survived.append(1 if stats.survived() else 0)

            mean_fitness = np.mean(fitness) if fitness else np.nan
            mean_distance = np.mean(distance) if distance else np.nan
            mean_variance = np.mean(variance) if variance else np.nan
            survival_rate = np.mean(survived) if survived else np.nan

            print(
                f"{condition:18s} | "
                f"fitness: {mean_fitness:.3f} | "
                f"dist: {mean_distance:.3f} | "
                f"var: {mean_variance:.4f} | "
                f"survival: {survival_rate:.2f}"
            )


if __name__ == "__main__":
    main()
