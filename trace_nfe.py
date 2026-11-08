import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D

class StrictCounter:
    def __init__(self, func):
        self.func = func
        self.actual_executions = 0
        self.wrapper_calls = 0

    def __call__(self, x):
        self.wrapper_calls += 1
        self.actual_executions += 1
        return self.func(x)

def trace_run(algo_name, b_name, seed=42):
    config = BENCHMARKS[b_name]
    rng = np.random.default_rng(seed)
    init_pop = rng.uniform(config["lb"], config["ub"], (N, D))
    counter = StrictCounter(config["func"])

    print(f"\n--- Tracing {algo_name} on {b_name} ---")

    if algo_name == "HAOA_orig":
        # T=167, 3N evals per iter = 90. 167 * 90 = 15030.
        opt = HAOAOriginal(counter, D, config["lb"], config["ub"], N, 167)
        opt.population = init_pop.copy()
        opt.optimize()
    elif algo_name == "HAOA_impr":
        # Init N (30) + T=500, N per iter = 15000. Total = 15030.
        opt = HAOAImproved(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
        opt.optimize()
    elif algo_name == "PSO":
        # Harness 30 + T=500, N per iter = 15000. Total = 15030.
        for p in init_pop: counter(p)
        opt = PSO(counter, D, config["lb"], config["ub"], N, 500)
        opt.positions = init_pop.copy()
        opt.personal_best_positions = init_pop.copy()
        opt.optimize()
    elif algo_name == "DE":
        # Init N (30) + T=500, N per iter = 15000. Total = 15030.
        opt = DifferentialEvolution(counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
        opt.optimize()

    print(f"Algorithm: {algo_name}")
    print(f"Actual Executions: {counter.actual_executions}")
    print(f"Wrapper Calls: {counter.wrapper_calls}")
    assert counter.actual_executions == 15030

if __name__ == "__main__":
    for algo in ["HAOA_orig", "HAOA_impr", "PSO", "DE"]:
        trace_run(algo, "Sphere")
