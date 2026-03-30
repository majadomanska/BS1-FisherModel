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
from experiment_flood import run_naive, run_pre_adapted

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

def summarize_stats(stats_list):
    fitness = []
    distance = []
    variance = []
    n_parents = []
    median_offspring = []
    max_offspring = []
    survived = []

    for stats in stats_list:
        if stats.records:
            last = stats.records[-1]

            fitness.append(last.mean_fitness)
            distance.append(last.distance_from_optimum)
            variance.append(last.phenotype_variance)

            if len(stats.n_parents_series) > 0:
                n_parents.append(np.mean(stats.n_parents_series))
            if len(stats.median_offspring_series) > 0:
                median_offspring.append(np.mean(stats.median_offspring_series))
            if len(stats.max_offspring_series) > 0:
                max_offspring.append(np.mean(stats.max_offspring_series))

            survived.append(1 if stats.survived() else 0)

    return {
        "mean_fitness": np.mean(fitness) if fitness else np.nan,
        "mean_distance": np.mean(distance) if distance else np.nan,
        "mean_variance": np.mean(variance) if variance else np.nan,
        "mean_n_parents": np.mean(n_parents) if n_parents else np.nan,
        "mean_median_offspring": np.mean(median_offspring) if median_offspring else np.nan,
        "mean_max_offspring": np.mean(max_offspring) if max_offspring else np.nan,
        "survival_rate": np.mean(survived) if survived else np.nan,
        "n_survived": int(np.sum(survived)) if survived else 0,
        "n_total": len(stats_list),
    }

def main():
    # two variants of parameters 
    variants = {
        "no_noise": 0.0,
        "with_noise": 0.02,
    }

    results = {}
    
    original_delta = config.delta
    
    for variant_name, delta_value in variants.items():
        print(f"\n=== {variant_name} | delta = {delta_value} ===")

        results[variant_name] = {
            "baseline": [],
            "naive_flood": [],
            "pre_adapted_flood": [],
        }

        for seed in range(20):
            stats_baseline = run_baseline(seed, delta_value)
            
            config.delta = delta_value
            try:
                stats_naive = run_naive(seed, frames_dir=None, verbose=False)
                stats_pre = run_pre_adapted(seed, frames_dir=None, verbose=False)
            finally:
                config.delta = original_delta

            results[variant_name]["baseline"].append(stats_baseline)
            results[variant_name]["naive_flood"].append(stats_naive)
            results[variant_name]["pre_adapted_flood"].append(stats_pre)

            print(f"done: {variant_name}, seed={seed}")
            
    config.delta = original_delta

    # SAVE FULL OBJECTS (pickle)
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\n Full results saved: results.pkl")

    # SAVE CSV
    with open("results_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "variant",
            "condition",
            "seed",
            "last_generation",
            "final_fitness",
            "final_distance",
            "final_variance",
            "mean_n_parents",
            "mean_median_offspring",
            "mean_max_offspring",
            "survived",
            "extinct_at",
        ])

        for variant_name, variant in results.items():
            for condition, stats_list in variant.items():
                for seed, stats in enumerate(stats_list):
                    if stats.records:
                        last = stats.records[-1]
                        
                        mean_n_parents = np.mean(stats.n_parents_series) if len(stats.n_parents_series) else np.nan
                        mean_median_offspring = np.mean(stats.median_offspring_series) if len(stats.median_offspring_series) else np.nan
                        mean_max_offspring = np.mean(stats.max_offspring_series) if len(stats.max_offspring_series) else np.nan
                        
                        writer.writerow([
                            variant_name,
                            condition,
                            seed,
                            last.generation,
                            last.mean_fitness,
                            last.distance_from_optimum,
                            last.phenotype_variance,
                            mean_n_parents,
                            mean_median_offspring,
                            mean_max_offspring,
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
                            None,
                            None,
                            None,
                            False,
                            stats.extinct_at,
                        ])

    print("Results saved: results_table.csv")
    
    #SUMMARY
    with open("summary_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant",
            "condition",
            "mean_fitness",
            "mean_distance",
            "mean_variance",
            "mean_n_parents",
            "mean_median_offspring",
            "mean_max_offspring",
            "survival_rate",
            "n_survived",
            "n_total",
        ])

        print("\n SUMMARY ")
        for variant_name, variant in results.items():
            print(f"\n--- {variant_name} ---")

            for condition, stats_list in variant.items():
                summary = summarize_stats(stats_list)

                writer.writerow([
                    variant_name,
                    condition,
                    summary["mean_fitness"],
                    summary["mean_distance"],
                    summary["mean_variance"],
                    summary["mean_n_parents"],
                    summary["mean_median_offspring"],
                    summary["mean_max_offspring"],
                    summary["survival_rate"],
                    summary["n_survived"],
                    summary["n_total"],
                ])

                print(
                    f"{condition:18s} | "
                    f"fitness: {summary['mean_fitness']:.3f} | "
                    f"dist: {summary['mean_distance']:.3f} | "
                    f"var: {summary['mean_variance']:.4f} | "
                    f"n_parents: {summary['mean_n_parents']:.2f} | "
                    f"med_off: {summary['mean_median_offspring']:.2f} | "
                    f"max_off: {summary['mean_max_offspring']:.2f} | "
                    f"survival: {summary['survival_rate']:.2f} "
                    f"({summary['n_survived']}/{summary['n_total']})"
                )

    print("\nSummary results saved: summary_results.csv")


if __name__ == "__main__":
    main()
