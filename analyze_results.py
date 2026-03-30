import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact


def pad_series(stats_list, attr):
    max_len = max(len(getattr(s, attr)) for s in stats_list)
    arr = np.full((len(stats_list), max_len), np.nan)

    for i, s in enumerate(stats_list):
        series = getattr(s, attr)
        arr[i, :len(series)] = series

    return arr


def final_values(stats_list, attr):
    vals = []
    for s in stats_list:
        series = getattr(s, attr)
        if len(series) > 0:
            vals.append(series[-1])
    return np.array(vals)


def extinction_counts(stats_list):
    extinct = sum(1 for s in stats_list if s.extinct_at is not None)
    survived = len(stats_list) - extinct
    return survived, extinct


def plot_mean_std(ax, stats_list, attr, label):
    arr = pad_series(stats_list, attr)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    x = np.arange(len(mean))

    ax.plot(x, mean, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)


def compare_extinction(stats_a, stats_b, label_a, label_b):
    surv_a, ext_a = extinction_counts(stats_a)
    surv_b, ext_b = extinction_counts(stats_b)

    table = [[ext_a, surv_a], [ext_b, surv_b]]
    oddsratio, pvalue = fisher_exact(table)

    print(f"\nExtinction comparison: {label_a} vs {label_b}")
    print(f"table = {table}")
    print(f"Fisher exact p-value = {pvalue:.6f}")


def main():
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)


    # Time-series plots: mean ± std

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes = axes.flatten()

    variants = list(results.keys())

    for i, variant_name in enumerate(variants):
        variant = results[variant_name]

        ax = axes[i]
        plot_mean_std(ax, variant["baseline"], "mean_fitnesses", "baseline")
        plot_mean_std(ax, variant["naive_flood"], "mean_fitnesses", "naive_flood")
        plot_mean_std(ax, variant["pre_adapted_flood"], "mean_fitnesses", "pre_adapted_flood")
        ax.set_title(f"Mean fitness ± std ({variant_name})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mean fitness")
        ax.legend()

    fig.tight_layout()
    fig.savefig("fitness_trajectories.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes = axes.flatten()

    for i, variant_name in enumerate(variants):
        variant = results[variant_name]

        ax = axes[i]
        plot_mean_std(ax, variant["baseline"], "distances_from_optimum", "baseline")
        plot_mean_std(ax, variant["naive_flood"], "distances_from_optimum", "naive_flood")
        plot_mean_std(ax, variant["pre_adapted_flood"], "distances_from_optimum", "pre_adapted_flood")
        ax.set_title(f"Distance from optimum ± std ({variant_name})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Distance from optimum")
        ax.legend()

    fig.tight_layout()
    fig.savefig("distance_trajectories.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes = axes.flatten()

    for i, variant_name in enumerate(variants):
        variant = results[variant_name]

        ax = axes[i]
        plot_mean_std(ax, variant["baseline"], "phenotype_variances", "baseline")
        plot_mean_std(ax, variant["naive_flood"], "phenotype_variances", "naive_flood")
        plot_mean_std(ax, variant["pre_adapted_flood"], "phenotype_variances", "pre_adapted_flood")
        ax.set_title(f"Phenotypic variance ± std ({variant_name})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Phenotypic variance")
        ax.legend()

    fig.tight_layout()
    fig.savefig("variance_trajectories.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes = axes.flatten()

    for i, variant_name in enumerate(variants):
        variant = results[variant_name]

        ax = axes[i]
        plot_mean_std(ax, variant["baseline"], "n_parents_series", "baseline: n_parents")
        plot_mean_std(ax, variant["naive_flood"], "n_parents_series", "naive_flood: n_parents")
        plot_mean_std(ax, variant["pre_adapted_flood"], "n_parents_series", "pre_adapted_flood: n_parents")
        ax.set_title(f"Reproductive inequality proxy ({variant_name})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("n_parents")
        ax.legend()

    fig.tight_layout()
    fig.savefig("reproductive_inequality.png")
    plt.close(fig)


    # Extinction rate bar plot

    conditions = ["baseline", "naive_flood", "pre_adapted_flood"]
    x = np.arange(len(variants))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, condition in enumerate(conditions):
        rates = []
        for variant_name in variants:
            stats_list = results[variant_name][condition]
            surv, ext = extinction_counts(stats_list)
            rates.append(ext / len(stats_list))
        ax.bar(x + i * width - width, rates, width, label=condition)

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Extinction rate")
    ax.set_title("Extinction rates by condition and parameter variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig("extinction_rates.png")
    plt.close(fig)


    # Final lag scatter

    fig, axes = plt.subplots(1, len(variants), figsize=(12, 5), sharey=True)

    if len(variants) == 1:
        axes = [axes]

    for ax, variant_name in zip(axes, variants):
        variant = results[variant_name]

        for condition in conditions:
            vals = final_values(variant[condition], "distances_from_optimum")
            ax.scatter([condition] * len(vals), vals, alpha=0.7)

        ax.set_title(f"Final lag from optimum ({variant_name})")
        ax.set_ylabel("Final distance")

    fig.tight_layout()
    fig.savefig("final_lag_scatter.png")
    plt.close(fig)


    # Mean fitness at generation 100 — parameter sensitivity

    target_gen = 100
    fig, ax = plt.subplots(figsize=(8, 5))

    for condition in conditions:
        means = []
        for variant_name in variants:
            arr = pad_series(results[variant_name][condition], "mean_fitnesses")
            if arr.shape[1] > target_gen:
                means.append(np.nanmean(arr[:, target_gen]))
            else:
                means.append(np.nan)
        ax.plot(variants, means, marker="o", label=condition)

    ax.set_title(f"Mean fitness at generation {target_gen}")
    ax.set_ylabel("Mean fitness")
    ax.set_xlabel("Parameter variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig("parameter_sensitivity_fitness.png")
    plt.close(fig)


    # Fisher exact tests

    for variant_name in variants:
        variant = results[variant_name]
        print("\n" + "=" * 60)
        print(f"VARIANT: {variant_name}")
        print("=" * 60)

        compare_extinction(
            variant["baseline"],
            variant["naive_flood"],
            "baseline",
            "naive_flood",
        )

        compare_extinction(
            variant["naive_flood"],
            variant["pre_adapted_flood"],
            "naive_flood",
            "pre_adapted_flood",
        )

        compare_extinction(
            variant["baseline"],
            variant["pre_adapted_flood"],
            "baseline",
            "pre_adapted_flood",
        )

    print("\nSaved all analysis plots.")


if __name__ == "__main__":
    main()
