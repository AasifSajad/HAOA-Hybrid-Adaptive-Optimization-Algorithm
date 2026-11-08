import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D, BUDGET

class CountingWrapper:
    def __init__(self, func):
        self.func = func
        self.total_calls = 0
    def __call__(self, x):
        self.total_calls += 1
        return self.func(x)

def run_pilot():
    results = []
    convergence_data = []
    output_dir = "results/pilot"
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/csv", exist_ok=True)
    seed = 42

    for b_name, config in BENCHMARKS.items():
        rng = np.random.default_rng(seed)
        init_pop = rng.uniform(config["lb"], config["ub"], (N, D))

        all_results_b = []

        for algo_name in ["HAOA_orig", "HAOA_impr", "PSO", "DE"]:
            counter = CountingWrapper(config["func"])
            np.random.seed(seed)

            if algo_name == "HAOA_orig":
                opt = HAOAOriginal(counter, D, config["lb"], config["ub"], N, 167)
                opt.population = init_pop.copy()
            elif algo_name == "HAOA_impr":
                opt = HAOAImproved(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)
            elif algo_name == "PSO":
                for p in init_pop: counter(p)
                opt = PSO(counter, D, config["lb"], config["ub"], N, 500)
                opt.positions = init_pop.copy()
                opt.personal_best_positions = init_pop.copy()
            elif algo_name == "DE":
                opt = DifferentialEvolution(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)

            start = time.time()
            res = opt.optimize()
            end = time.time()

            assert counter.total_calls == BUDGET, f"{algo_name} budget fail: {counter.total_calls}"

            results.append({
                "Algorithm": algo_name, "Benchmark": b_name, "Score": res["best_score"],
                "Runtime": end - start, "NFE": counter.total_calls, "Termination": "Completed"
            })

            all_results_b.append({"name": algo_name, "curve": res["convergence_curve"]})

            for step, val in enumerate(res["convergence_curve"]):
                convergence_data.append({
                    "Algorithm": algo_name, "Benchmark": b_name, "Step": step + 1, "BestScore": val
                })

        # Plot convergence
        plt.figure(figsize=(10, 6))
        for item in all_results_b:
            plt.plot(item["curve"], label=item["name"], linewidth=2)
        plt.title(f"{b_name} Pilot Convergence (Seed {seed})")
        plt.xlabel("Step / Iteration")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/{b_name}_pilot.png")
        plt.close()

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{output_dir}/pilot_results.csv", index=False)

    df_conv = pd.DataFrame(convergence_data)
    df_conv.to_csv(f"{output_dir}/csv/pilot_convergence.csv", index=False)

    print("Pilot Results CSV:")
    print(df_res.to_string())

if __name__ == "__main__":
    run_pilot()
