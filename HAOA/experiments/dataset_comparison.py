"""Paired multi-seed HAOA-versus-PSO benchmark experiments."""
import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from HAOA.algorithm.haoa import HAOA
from HAOA.comparison_algorithms.pso import PSO
from HAOA.experiments.benchmark_manager import BenchmarkManager


class SimplePSO(PSO):
    """Backward-compatible alias for the shared PSO implementation."""


def _bootstrap_median_ci(differences, seed=2026, samples=2000):
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("at least one paired difference is required")
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, size=(samples, values.size), replace=True), axis=1)
    return tuple(float(x) for x in np.quantile(medians, [0.025, 0.975]))


def summarize_paired_scores(haoa_scores, pso_scores):
    haoa, pso = np.asarray(haoa_scores, dtype=float), np.asarray(pso_scores, dtype=float)
    if haoa.ndim != 1 or pso.ndim != 1 or haoa.shape != pso.shape or haoa.size == 0:
        raise ValueError("HAOA and PSO scores must be non-empty paired 1-D arrays")
    if not np.all(np.isfinite(haoa)) or not np.all(np.isfinite(pso)):
        raise ValueError("scores must all be finite")
    diff = haoa - pso
    low, high = _bootstrap_median_ci(diff)
    return {
        "runs": int(haoa.size),
        "haoa_mean": float(np.mean(haoa)),
        "haoa_std": float(np.std(haoa, ddof=1)) if haoa.size > 1 else 0.0,
        "haoa_median": float(np.median(haoa)),
        "pso_mean": float(np.mean(pso)),
        "pso_std": float(np.std(pso, ddof=1)) if pso.size > 1 else 0.0,
        "pso_median": float(np.median(pso)),
        "haoa_wins": int(np.sum(diff < 0)),
        "pso_wins": int(np.sum(diff > 0)),
        "ties": int(np.sum(diff == 0)),
        "median_paired_difference": float(np.median(diff)),
        "paired_median_ci_2_5": low,
        "paired_median_ci_97_5": high,
    }


def run_dataset_test(*, seeds=None, dimension=30, population_size=30,
                     max_iterations=100, output_dir=None):
    if seeds is None:
        seeds = list(range(30))
    else:
        seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    output = Path(output_dir) if output_dir is not None else Path("HAOA/results")
    output.mkdir(parents=True, exist_ok=True)
    raw_rows, summary_rows = [], []

    for name, config in BenchmarkManager().get_all_benchmarks().items():
        function = config["function"]
        lower, upper = config["lower_bound"], config["upper_bound"]
        haoa_scores, pso_scores, haoa_curves, pso_curves = [], [], [], []
        for seed in seeds:
            haoa = HAOA(function, dimension, lower, upper, population_size,
                        max_iterations, random_state=seed)
            start = time.perf_counter()
            hr = haoa.optimize()
            ht = time.perf_counter() - start

            pso = PSO(function, dimension, lower, upper, population_size,
                      max_iterations, random_state=seed)
            start = time.perf_counter()
            pr = pso.optimize()
            pt = time.perf_counter() - start

            hs, ps = float(hr["best_score"]), float(pr["best_score"])
            if not np.isfinite(hs) or not np.isfinite(ps):
                raise RuntimeError(f"{name}: non-finite score for seed {seed}")
            haoa_scores.append(hs)
            pso_scores.append(ps)
            haoa_curves.append(hr["convergence_curve"])
            pso_curves.append(pr["convergence_curve"])
            raw_rows.append({
                "Benchmark": name, "Seed": seed, "HAOA Score": hs, "PSO Score": ps,
                "HAOA Time Seconds": ht, "PSO Time Seconds": pt,
                "Paired Difference HAOA_minus_PSO": hs - ps,
            })

        stats = summarize_paired_scores(haoa_scores, pso_scores)
        summary_rows.append({"Benchmark": name, **stats})
        plt.figure(figsize=(9, 5))
        plt.plot(np.mean(np.asarray(haoa_curves), axis=0), label="HAOA mean best-so-far")
        plt.plot(np.mean(np.asarray(pso_curves), axis=0), label="PSO mean best-so-far")
        plt.xlabel("Iteration")
        plt.ylabel("Mean best-so-far fitness")
        plt.title(f"{name}: paired multi-seed convergence (n={len(seeds)})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output / f"{name}_comparison.png", dpi=140)
        plt.close()

    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "benchmark_comparison_runs.csv", raw_rows)
    write_csv(output / "benchmark_comparison_summary.csv", summary_rows)
    with (output / "benchmark_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Benchmark", "HAOA Score", "PSO Score", "HAOA Time", "PSO Time",
                         "Runs", "HAOA Wins", "PSO Wins", "Ties", "Median Paired Difference",
                         "Paired Median CI Lower", "Paired Median CI Upper"])
        for row in summary_rows:
            subset = [item for item in raw_rows if item["Benchmark"] == row["Benchmark"]]
            writer.writerow([row["Benchmark"], row["haoa_mean"], row["pso_mean"],
                np.mean([x["HAOA Time Seconds"] for x in subset]),
                np.mean([x["PSO Time Seconds"] for x in subset]), row["runs"],
                row["haoa_wins"], row["pso_wins"], row["ties"],
                row["median_paired_difference"], row["paired_median_ci_2_5"],
                row["paired_median_ci_97_5"]])
    print(f"Completed {len(summary_rows)} benchmarks x {len(seeds)} paired seeds.")
    print(f"Raw runs: {output / 'benchmark_comparison_runs.csv'}")
    print(f"Aggregate statistics: {output / 'benchmark_comparison_summary.csv'}")
    for row in summary_rows:
        print(f"{row['Benchmark']}: median HAOA={row['haoa_median']:.6g}, "
              f"median PSO={row['pso_median']:.6g}; wins HAOA/PSO/tie="
              f"{row['haoa_wins']}/{row['pso_wins']}/{row['ties']}; "
              f"paired median difference={row['median_paired_difference']:.6g}; "
              f"95% bootstrap CI=[{row['paired_median_ci_2_5']:.6g}, "
              f"{row['paired_median_ci_97_5']:.6g}]")
    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--dimension", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    run_dataset_test(seeds=range(args.seed_start, args.seed_start + args.runs),
                     dimension=args.dimension, population_size=args.population_size,
                     max_iterations=args.iterations, output_dir=args.output_dir)
