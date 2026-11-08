import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution

from HAOA.benchmark_functions.sphere import sphere
from HAOA.benchmark_functions.rastrigin import rastrigin
from HAOA.benchmark_functions.rosenbrock import rosenbrock
from HAOA.benchmark_functions.ackley import ackley
from HAOA.benchmark_functions.griewank import griewank

# Standard Benchmark Configuration with exact standard bounds
BENCHMARKS = {
    "Sphere": {"func": sphere, "lb": -100, "ub": 100},
    "Rastrigin": {"func": rastrigin, "lb": -5.12, "ub": 5.12},
    "Rosenbrock": {"func": rosenbrock, "lb": -30, "ub": 30},
    "Ackley": {"func": ackley, "lb": -32.768, "ub": 32.768},
    "Griewank": {"func": griewank, "lb": -600, "ub": 600}
}

N = 30
D = 30
BUDGET = 15030

class ExperimentHarness:
    def __init__(self, output_dir="results/phase1"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

    def get_initial_population(self, seed, benchmark_name):
        config = BENCHMARKS[benchmark_name]
        rng = np.random.default_rng(seed)
        return rng.uniform(config["lb"], config["ub"], (N, D))

    def run_single(self, algo_name, benchmark_name, seed):
        config = BENCHMARKS[benchmark_name]
        init_pop = self.get_initial_population(seed, benchmark_name)

        # Exact NFE budget matching (= 15,030)
        if algo_name == "HAOA_orig":
            # 3 * N * T = 3 * 30 * 167 = 15,030
            optimizer = HAOAOriginal(
                objective_function=config["func"],
                dimension=D,
                lower_bound=config["lb"],
                upper_bound=config["ub"],
                population_size=N,
                max_iterations=167
            )
            optimizer.population = init_pop.copy()

        elif algo_name == "HAOA_impr":
            # N + N * T = 30 + 30 * 500 = 15,030
            optimizer = HAOAImproved(
                objective_function=config["func"],
                dimension=D,
                lower_bound=config["lb"],
                upper_bound=config["ub"],
                population_size=N,
                max_iterations=500,
                initial_population=init_pop,
                random_seed=seed,
                evaluations_limit=BUDGET
            )
        elif algo_name == "PSO":
            # N + N * T = 30 + 30 * 500 = 15,030
            # Pre-evaluate initial population (30 evals) via wrapper or explicit call
            def pso_obj_wrapper(x):
                return config["func"](x)

            for p in init_pop:
                pso_obj_wrapper(p)

            optimizer = PSO(
                objective_function=pso_obj_wrapper,
                dimension=D,
                lower_bound=config["lb"],
                upper_bound=config["ub"],
                population_size=N,
                max_iterations=500
            )
            optimizer.positions = init_pop.copy()
            optimizer.personal_best_positions = init_pop.copy()

        elif algo_name == "DE":
            # N + N * G = 30 + 30 * 500 = 15,030
            optimizer = DifferentialEvolution(
                objective_function=config["func"],
                dimension=D,
                lower_bound=config["lb"],
                upper_bound=config["ub"],
                population_size=N,
                max_generations=500,
                initial_population=init_pop,
                random_seed=seed,
                evaluations_limit=BUDGET
            )

        start_time = time.time()
        result = optimizer.optimize()
        runtime = time.time() - start_time

        return {
            "score": result["best_score"],
            "runtime": runtime,
            "curve": result["convergence_curve"]
        }

if __name__ == "__main__":
    print("Experiment Harness Updated with corrected PSO iteration count (T=500).")
