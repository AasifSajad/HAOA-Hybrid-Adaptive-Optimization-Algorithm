import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D

class MetricsCounter:
    def __init__(self, func):
        self.func = func
        self.total_calls = 0
    def __call__(self, x):
        self.total_calls += 1
        return self.func(x)

def verify_nfe_logic():
    seed = 42
    b_name = "Sphere"
    config = BENCHMARKS[b_name]
    rng = np.random.default_rng(seed)
    init_pop = rng.uniform(config["lb"], config["ub"], (N, D))

    print(f"{'Algorithm':<12} | {'Init Evals':<10} | {'Iter Evals':<10} | {'Iterations':<10} | {'Calculated Total'}")
    print("-" * 75)

    # PSO Analysis:
    # PSO class does NOT eval initial population.
    # Harness does N evals (30).
    # PSO optimize() does N * iterations.
    # To get 15030 total: 15030 = 30 + N * T => 15000 = 30 * T => T = 500.

    # Correction: PSO should run for 500 iterations, not 501.
    pso_counter = MetricsCounter(config["func"])
    for p in init_pop: pso_counter(p) # 30
    pso = PSO(pso_counter, D, config["lb"], config["ub"], N, 500)
    pso.positions = init_pop.copy()
    pso.personal_best_positions = init_pop.copy()
    pso.optimize()
    print(f"{'PSO':<12} | {30:<10} | {30:<10} | {500:<10} | {30 + 500*30}")

    # HAOA Original
    # 167 iterations. 3N per iteration = 90. 167*90 = 15030.
    haoa_counter = MetricsCounter(config["func"])
    opt = HAOAOriginal(haoa_counter, D, config["lb"], config["ub"], N, 167)
    opt.population = init_pop.copy()
    opt.optimize()
    print(f"{'HAOA_orig':<12} | {0:<10} | {90:<10} | {167:<10} | {167*90}")

    # HAOA Improved
    # Init 30 + 500 iterations * 30 = 15030.
    haoa_i_counter = MetricsCounter(config["func"])
    opt = HAOAImproved(haoa_i_counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
    opt.optimize()
    print(f"{'HAOA_impr':<12} | {30:<10} | {30:<10} | {500:<10} | {30 + 500*30}")

    # DE
    # Init 30 + 500 gens * 30 = 15030.
    de_counter = MetricsCounter(config["func"])
    opt = DifferentialEvolution(de_counter, D, config["lb"], config["ub"], N, 500, initial_population=init_pop, random_seed=seed, evaluations_limit=15030)
    opt.optimize()
    print(f"{'DE':<12} | {30:<10} | {30:<10} | {500:<10} | {30 + 500*30}")

verify_nfe_logic()
