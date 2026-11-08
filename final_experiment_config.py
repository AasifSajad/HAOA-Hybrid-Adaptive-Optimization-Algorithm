import os
import time
import json
import numpy as np
import pandas as pd
from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.algorithm.haoa_advanced import HAOAAdvanced
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.benchmark_functions.sphere import sphere
from HAOA.benchmark_functions.rastrigin import rastrigin
from HAOA.benchmark_functions.rosenbrock import rosenbrock
from HAOA.benchmark_functions.ackley import ackley
from HAOA.benchmark_functions.griewank import griewank

# --- Configuration ---
BENCHMARKS = {
    "Sphere": {"func": sphere, "lb": -100, "ub": 100},
    "Rastrigin": {"func": rastrigin, "lb": -5.12, "ub": 5.12},
    "Rosenbrock": {"func": rosenbrock, "lb": -30, "ub": 30},
    "Ackley": {"func": ackley, "lb": -32.768, "ub": 32.768},
    "Griewank": {"func": griewank, "lb": -600, "ub": 600}
}
ALGORITHMS = ["HAOA_orig", "HAOA_impr", "HAOA_adv", "PSO", "DE"]
N = 30
D = 30
BUDGET = 15030
SEEDS = list(range(100, 130)) # 30 independent seeds, distinct from previous tests

OUTPUT_DIR = "results/final_experiment"
RAW_DATA_FILE = os.path.join(OUTPUT_DIR, "raw_results.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

# --- Transparent Counter ---
class TransparentCounter:
    def __init__(self, func, budget=15030):
        self.func = func
        self.budget = budget
        self.actual_executions = 0
    def __call__(self, x):
        self.actual_executions += 1
        return self.func(x)

# --- Bounded Wrappers for strict NFE termination ---
def get_optimizer(algo_name, config, counter, init_pop, seed):
    if algo_name == "HAOA_orig":
        class BoundedHAOA(HAOAOriginal):
            def optimize(self):
                for iteration in range(self.max_iterations):
                    if self.objective_function.actual_executions >= BUDGET: break
                    self.evaluate_population()
                    if self.objective_function.actual_executions >= BUDGET: break
                    self.update_population(iteration)
                    self.convergence_curve.append(self.best_score)
                return {"best_score": self.best_score, "curve": self.convergence_curve}
        opt = BoundedHAOA(counter, D, config["lb"], config["ub"], N, 167)
        opt.population = init_pop.copy()
        return opt

    elif algo_name == "HAOA_impr":
        class BoundedHAOAImpr(HAOAImproved):
            def update_population(self, iteration):
                for i in range(self.population_size):
                    if self.objective_function.actual_executions >= BUDGET: break
                    # ... core logic implementation ...
                    # (Standard HAOAImproved update logic omitted for brevity in config preview,
                    # full script will have exact frozen logic)
                    pass
        # Note: In the actual runner, we use the subclass pattern to enforce the budget check mid-iteration
        return HAOAImproved(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=BUDGET)

    # (Other algorithms similarly wrapped...)
    return None

def main():
    print(f"Final Experiment Configuration:")
    print(f"- Total Algos: {len(ALGORITHMS)} ({', '.join(ALGORITHMS)})")
    print(f"- Total Benchmarks: {len(BENCHMARKS)} ({', '.join(BENCHMARKS.keys())})")
    print(f"- Seeds: {len(SEEDS)} ({SEEDS[0]} to {SEEDS[-1]})")
    print(f"- Total Runs: {len(ALGORITHMS) * len(BENCHMARKS) * len(SEEDS)}")
    print(f"- Budget: {BUDGET} actual evaluations per run")
    print(f"- Dimension: {D}")
    print(f"- Population: {N}")

if __name__ == "__main__":
    main()
