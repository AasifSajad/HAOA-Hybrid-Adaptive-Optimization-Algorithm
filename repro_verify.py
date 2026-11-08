import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from HAOA.algorithm.haoa import HAOA as HAOAOriginal
from HAOA.algorithm.haoa_improved import HAOAImproved
from HAOA.comparison_algorithms.pso import PSO
from HAOA.comparison_algorithms.de import DifferentialEvolution
from HAOA.experiments.phase1_experiment import BENCHMARKS, N, D

def test_reproducibility():
    for b_name in BENCHMARKS:
        config = BENCHMARKS[b_name]
        seed = 42
        results = {"HAOA_orig": [], "HAOA_impr": [], "PSO": [], "DE": []}

        for i in range(3):
            # Seed global RNG for original algos
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            init_pop = rng.uniform(config["lb"], config["ub"], (N, D))

            # HAOA_orig
            np.random.seed(seed)
            opt = HAOAOriginal(config["func"], D, config["lb"], config["ub"], N, 10)
            opt.population = init_pop.copy()
            results["HAOA_orig"].append(opt.optimize()["best_score"])

            # HAOA_impr
            opt = HAOAImproved(config["func"], D, config["lb"], config["ub"], N, 10, initial_population=init_pop, random_seed=seed)
            results["HAOA_impr"].append(opt.optimize()["best_score"])

            # PSO
            np.random.seed(seed)
            opt = PSO(config["func"], D, config["lb"], config["ub"], N, 10)
            opt.positions = init_pop.copy()
            opt.personal_best_positions = init_pop.copy()
            results["PSO"].append(opt.optimize()["best_score"])

            # DE
            opt = DifferentialEvolution(config["func"], D, config["lb"], config["ub"], N, 10, initial_population=init_pop, random_seed=seed)
            results["DE"].append(opt.optimize()["best_score"])

        print(f"\n--- {b_name} Reproducibility (with global seed reset) ---")
        for algo in results:
            unique_vals = set(results[algo])
            status = "PASSED" if len(unique_vals) == 1 else f"FAILED: {results[algo]}"
            print(f"{algo:<12}: {status}")

if __name__ == "__main__":
    test_reproducibility()
