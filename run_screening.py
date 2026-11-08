import numpy as np
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.algorithm.haoa_advanced import HAOAAdvanced
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D, BUDGET
from run_corrected_pilot import DetailedCounter

def run_screening():
    output_dir = "results/advanced_screening"
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)

    seeds = [42, 43, 44, 45, 46]
    raw_results = []

    # Store curves for seed 42 plotting
    curves_seed_42 = {b: [] for b in BENCHMARKS}
    diversity_seed_42 = {b: [] for b in BENCHMARKS}

    # Silence verbose print from HAOAOriginal
    old_stdout = sys.stdout

    for b_name, config in BENCHMARKS.items():
        for seed in seeds:
            rng = np.random.default_rng(seed)
            init_pop = rng.uniform(config["lb"], config["ub"], (N, D))

            for algo_name in ["HAOA_orig", "HAOA_impr", "HAOA_adv"]:
                counter = DetailedCounter(config["func"])
                np.random.seed(seed)

                # Suppress output during opt
                sys.stdout = open(os.devnull, 'w')
                start_time = time.time()

                if algo_name == "HAOA_orig":
                    opt = HAOAOriginal(counter, D, config["lb"], config["ub"], N, 167)
                    opt.population = init_pop.copy()
                    res = opt.optimize()
                    diversity = [0.0] * len(res["convergence_curve"]) # orig doesn't track diversity
                elif algo_name == "HAOA_impr":
                    opt = HAOAImproved(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
                    res = opt.optimize()
                    diversity = res.get("diversity_curve", [0.0] * len(res["convergence_curve"]))
                elif algo_name == "HAOA_adv":
                    opt = HAOAAdvanced(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
                    res = opt.optimize()
                    diversity = res.get("diversity_curve", [0.0] * len(res["convergence_curve"]))

                runtime = time.time() - start_time
                sys.stdout = old_stdout

                raw_results.append({
                    "Algorithm": algo_name,
                    "Benchmark": b_name,
                    "Seed": seed,
                    "Requests": counter.total_requests,
                    "ActualExecutions": counter.actual_executions,
                    "CacheHits": counter.cache_hits,
                    "BestFitness": res["best_score"],
                    "Runtime": runtime
                })

                if seed == 42:
                    curves_seed_42[b_name].append({
                        "name": algo_name,
                        "curve": res["convergence_curve"]
                    })
                    diversity_seed_42[b_name].append({
                        "name": algo_name,
                        "curve": diversity
                    })

    df = pd.DataFrame(raw_results)
    df.to_csv(f"{output_dir}/screening_raw_results.csv", index=False)

    # Summary Statistics Table
    summary = []
    for b_name in BENCHMARKS:
        for algo_name in ["HAOA_orig", "HAOA_impr", "HAOA_adv"]:
            sub_df = df[(df["Benchmark"] == b_name) & (df["Algorithm"] == algo_name)]
            scores = sub_df["BestFitness"].values
            runtimes = sub_df["Runtime"].values
            nfe_execs = sub_df["ActualExecutions"].values

            summary.append({
                "Benchmark": b_name,
                "Algorithm": algo_name,
                "Best": np.min(scores),
                "Median": np.median(scores),
                "Mean": np.mean(scores),
                "Std": np.std(scores),
                "MeanRuntime": np.mean(runtimes),
                "ExactNFE": int(np.mean(nfe_execs))
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/tables/screening_summary.csv", index=False)

    # Plot Convergence and Diversity Curves for Seed 42
    for b_name in BENCHMARKS:
        # Convergence
        plt.figure(figsize=(9, 5))
        for item in curves_seed_42[b_name]:
            plt.plot(item["curve"], label=item["name"], linewidth=2)
        plt.title(f"{b_name} Convergence Comparison (Seed 42)")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (Log Scale)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/{b_name}_convergence.png")
        plt.close()

        # Diversity
        plt.figure(figsize=(9, 5))
        for item in diversity_seed_42[b_name]:
            if item["name"] != "HAOA_orig":
                plt.plot(item["curve"], label=item["name"], linewidth=2)
        plt.title(f"{b_name} Population Diversity Comparison (Seed 42)")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Pairwise Distance")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/{b_name}_diversity.png")
        plt.close()

    print("Screening script executed successfully.")

if __name__ == "__main__":
    run_screening()
