import numpy as np
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D, BUDGET

class DetailedCounter:
    def __init__(self, func):
        self.func = func
        self.total_requests = 0
        self.actual_executions = 0
        self.cache_hits = 0
        self.seen_hashes = set()

    def __call__(self, x):
        self.total_requests += 1
        h = hash(x.tobytes())
        if h in self.seen_hashes:
            self.cache_hits += 1
        else:
            self.seen_hashes.add(h)
            self.actual_executions += 1
        return self.func(x)

def run_corrected_pilot():
    output_dir = "results/pilot_corrected"
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    seed = 42
    results = []

    print(f"{'Algorithm':<12} | {'Benchmark':<12} | {'Requests':<8} | {'Executions':<10} | {'Cache Hits':<10} | {'Best Score':<15}")
    print("-" * 75)

    for b_name, config in BENCHMARKS.items():
        rng = np.random.default_rng(seed)
        init_pop = rng.uniform(config["lb"], config["ub"], (N, D))

        all_curves = []

        for algo_name in ["HAOA_orig", "HAOA_impr", "PSO", "DE"]:
            counter = DetailedCounter(config["func"])
            np.random.seed(seed)

            if algo_name == "HAOA_orig":
                opt = HAOAOriginal(counter, D, config["lb"], config["ub"], N, 167)
                opt.population = init_pop.copy()
            elif algo_name == "HAOA_impr":
                opt = HAOAImproved(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
            elif algo_name == "PSO":
                for p in init_pop:
                    counter(p) # init
                opt = PSO(counter, D, config["lb"], config["ub"], N, 500)
                opt.positions = init_pop.copy()
                opt.personal_best_positions = init_pop.copy()
            elif algo_name == "DE":
                opt = DifferentialEvolution(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)

            start = time.time()
            res = opt.optimize()
            runtime = time.time() - start

            results.append({
                "Algorithm": algo_name,
                "Benchmark": b_name,
                "Seed": seed,
                "Requests": counter.total_requests,
                "ActualExecutions": counter.actual_executions,
                "CacheHits": counter.cache_hits,
                "BestFitness": res["best_score"],
                "Runtime": runtime,
                "Termination": "Completed"
            })

            print(f"{algo_name:<12} | {b_name:<12} | {counter.total_requests:<8} | {counter.actual_executions:<10} | {counter.cache_hits:<10} | {res['best_score']:<15.5e}")
            all_curves.append({"name": algo_name, "curve": res["convergence_curve"]})

        # Plot
        plt.figure(figsize=(10, 6))
        for item in all_curves:
            plt.plot(item["curve"], label=item["name"], linewidth=2)
        plt.title(f"{b_name} Corrected Pilot Convergence (Seed {seed})")
        plt.xlabel("Step / Iteration")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/{b_name}_corrected_pilot.png")
        plt.close()

    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/pilot_results_corrected.csv", index=False)

    # Print HAOA vs PSO comparison table
    print("\n" + "="*50)
    print(" HAOA (Improved) vs PSO Comparison ")
    print("="*50)
    print(f"{'Benchmark':<12} | {'HAOA Imp. Score':<15} | {'PSO Score':<15} | {'Winner'}")
    print("-" * 60)
    for b_name in BENCHMARKS:
        haoa_score = df[(df["Benchmark"] == b_name) & (df["Algorithm"] == "HAOA_impr")]["BestFitness"].values[0]
        pso_score = df[(df["Benchmark"] == b_name) & (df["Algorithm"] == "PSO")]["BestFitness"].values[0]
        winner = "HAOA" if haoa_score < pso_score else "PSO"
        print(f"{b_name:<12} | {haoa_score:<15.5e} | {pso_score:<15.5e} | {winner}")

if __name__ == "__main__":
    run_corrected_pilot()
